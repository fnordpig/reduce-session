"""Session discovery and metadata extraction for Claude Code sessions.

Scans project directories for session JSONL files, parses conversation tails
for preview, and extracts metadata. Used by the TUI to populate the session
tree and preview pane. Pure data module with no textual dependency.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from .session_formats import detect_codec
from .typing_aliases import BlockType

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
CONTINUATION_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.\d+.*\.jsonl$"
)
TIMESTAMP_CONTINUATION_RE = re.compile(
    r"^(.+)\.\d+.*\.jsonl$"
)
ROLLOUT_RE = re.compile(r"^rollout-(.+)\.jsonl$")
ROLLOUT_ID_RE = re.compile(
    r"^rollout-.*-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl$"
)
UUID_IN_NAME_RE = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})(?:\.jsonl)?$"
)
SKIP_SUFFIXES = (".bak", ".bak2", ".reduced")
SKIP_TYPES = frozenset({"progress", "system", "file-history-snapshot", "last-prompt"})


@dataclass
class Exchange:
    role: str  # "user", "assistant", "tool"
    text: str
    tool_name: str | None = None
    tool_status: str | None = None
    is_error: bool = False


@dataclass
class SessionInfo:
    path: Path
    project_name: str
    session_id: str  # full UUID
    short_id: str  # first 8 chars
    size_bytes: int
    token_estimate: int
    last_timestamp: datetime | None
    age_display: str  # "4h", "2d", "14d"
    line_count: int
    continuation_files: list[Path] = field(default_factory=list)
    last_exchanges: list[Exchange] = field(default_factory=list)
    parse_error: bool = False
    density_profile: list[int] = field(default_factory=list)
    resolved_dir: Path | None = None  # actual project directory on disk
    is_dangling: bool = False  # True if slug doesn't match a real directory
    project_slug: str = ""  # raw directory slug for full path display
    provider: str = "claude"
    branch: str | None = None
    branch_last_timestamp: datetime | None = None
    source_format: str = "claude"
    source_id: str = ""
    source_path: Path | None = None
    session_title: str | None = None

    def all_paths(self, *, sort_by_mtime: bool = False) -> list[Path]:
        """Unique paths for this session, including continuation files.

        ``sort_by_mtime=True`` orders newest-first then dedupes
        (used by the Reduce modal so replay preserves the conversational
        timeline). ``False`` preserves insertion order (used by the
        History modal which lists chronologically)."""
        paths: list[Path] = [self.path, *self.continuation_files]
        if sort_by_mtime:
            paths.sort(
                key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
                reverse=True,
            )
        seen: set[Path] = set()
        deduped: list[Path] = []
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            deduped.append(p)
        if sort_by_mtime:
            # Re-sort ascending so replay reads oldest → newest.
            deduped.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
        return deduped

    def merged_jsonl_tempfile(self) -> str | None:
        """Concatenate all session parts into a temp JSONL file.

        Returns the path. If there's only one part, returns that path
        directly (no temp file needed). Returns None if no readable lines
        are found. Caller is responsible for cleanup via
        :meth:`cleanup_merged_tempfile`."""
        import tempfile

        paths = self.all_paths(sort_by_mtime=True)
        all_lines: list[str] = []
        for p in paths:
            try:
                with p.open("r", errors="replace") as f:
                    for line in f:
                        if line:
                            all_lines.append(line)
            except OSError:
                continue
        if not all_lines:
            return None
        if len(paths) == 1:
            return str(paths[0])
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tf:
            tf.writelines(all_lines)
            return tf.name

    def cleanup_merged_tempfile(self, path: str | None) -> None:
        """Delete the temp file produced by :meth:`merged_jsonl_tempfile`.

        Idempotent and never-raises. If ``path`` equals the original
        session path (single-part session), nothing is deleted."""
        if path is None or path == str(self.path):
            return
        try:
            import os as _os

            _os.unlink(path)
        except OSError:
            pass


def resolve_slug_to_path(slug: str) -> Path | None:
    """Resolve a Claude Code project slug to the actual filesystem directory.

    Claude Code encodes /Users/rwaugh/src/mine/ripvec as
    -Users-rwaugh-src-mine-ripvec. Since path components can contain
    hyphens, we walk the filesystem: start from /, progressively join
    one segment at a time, and when a segment doesn't exist as a dir,
    try joining it with the next segment via hyphen until we find a
    match or exhaust the slug.

    Returns the resolved Path, or None if no matching directory exists.
    """
    if not slug.startswith("-"):
        return None

    segments = slug[1:].split("-")
    if not segments:
        return None

    current = Path("/")
    i = 0
    while i < len(segments):
        # Try progressively joining segments with hyphens
        found = False
        for j in range(i + 1, len(segments) + 1):
            candidate = "-".join(segments[i:j])
            test_path = current / candidate
            if test_path.is_dir():
                current = test_path
                i = j
                found = True
                break
        if not found:
            # No directory matches from position i — slug is dangling
            return None

    return current


# Cache for slug resolution (populated once per scan)
_slug_cache: dict[str, tuple[Path | None, str]] = {}


def derive_project_name(slug: str) -> str:
    """Convert directory slug to readable name.

    Resolves the slug to an actual filesystem path, then derives a
    short display name from it. Caches results.

    '-Users-rwaugh-src-mine-ripvec' -> 'ripvec'
    '-Users-rwaugh-src-archiuvium-aws-org-root' -> 'archiuvium/aws-org-root'
    (dangling slug) -> '? slug-text'
    """
    if slug in _slug_cache:
        return _slug_cache[slug][1]

    resolved = resolve_slug_to_path(slug)
    if resolved is None:
        # No matching directory — extract best name from slug pattern
        # Still useful for test fixtures and deleted projects
        parts = [p for p in slug.split("-") if p]
        # Skip home dir components
        skip = {"Users", "home"}
        tail = []
        found_user = False
        for p in parts:
            if p in skip:
                found_user = True
                tail.clear()
                continue
            if found_user and not tail:
                # This is the username — skip it
                found_user = False
                continue
            if p == "src" and not tail:
                continue
            if p == "mine" and not tail:
                continue
            tail.append(p)
        name = "-".join(tail) if tail else (parts[-1] if parts else slug)
        _slug_cache[slug] = (None, name)
        return name

    # Derive a short name from the resolved path
    home = Path.home()
    try:
        rel = resolved.relative_to(home)
    except ValueError:
        rel = resolved

    parts = rel.parts
    # Strip common prefixes: src, src/mine
    interesting = list(parts)
    for prefix in ("src", "mine"):
        if interesting and interesting[0] == prefix:
            interesting.pop(0)

    name = "/".join(interesting) if interesting else resolved.name
    _slug_cache[slug] = (resolved, name)
    return name


def get_resolved_path(slug: str) -> Path | None:
    """Get the resolved filesystem path for a slug (from cache)."""
    if slug not in _slug_cache:
        derive_project_name(slug)  # populates cache
    return _slug_cache[slug][0]


def is_dangling_project(slug: str) -> bool:
    """Check if a project slug doesn't map to a real directory."""
    return get_resolved_path(slug) is None


def format_age(timestamp: datetime) -> str:
    """Format a timestamp as relative age: '4m', '2h', '3d', '14d'."""
    now = datetime.now(timezone.utc)
    # Handle naive datetimes by assuming UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    delta = now - timestamp
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "0m"
    minutes = total_seconds // 60
    hours = total_seconds // 3600
    days = total_seconds // 86400
    if days >= 1:
        return f"{days}d"
    if hours >= 1:
        return f"{hours}h"
    return f"{minutes}m"


def _parse_timestamp(ts_str: str | None) -> datetime | None:
    """Parse an ISO timestamp string, returning None on failure."""
    if not ts_str:
        return None
    try:
        # Handle Z suffix
        s = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _coerce_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text if text else None


def _ensure_aware_utc(ts: datetime | None) -> datetime | None:
    """Return timestamp with UTC tzinfo, preserving existing timezone info."""
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def _normalize_session_record(record: object) -> dict[str, object] | None:
    """Normalize format-specific session records into a common structure."""
    if not isinstance(record, dict):
        return None
    try:
        typed_record = cast(dict[str, Any], record)
        codec = detect_codec([typed_record])
        normalized = codec.normalize(typed_record)
        if isinstance(normalized, dict):
            return cast(dict[str, object], normalized)
    except Exception:
        return None


def _to_dict_payload(payload: object) -> dict[str, Any] | None:
    """Convert a payload-like object into a strongly-typed string-key dict."""
    if not isinstance(payload, dict):
        return None
    return cast(dict[str, Any], payload)


def _extract_message_like_block(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract message-like metadata from an arbitrary payload container."""
    msg_field = payload.get("message")
    if isinstance(msg_field, dict):
        return cast(dict[str, Any], msg_field)

    msg = payload.get("message")
    if msg is not None:
        role = payload.get("role", "assistant")
        if not isinstance(role, str):
            role = "assistant"
        return {"content": msg, "role": role}

    if "content" in payload:
        content = payload.get("content")
        if content is not None:
            role = payload.get("role", "assistant")
            if not isinstance(role, str):
                role = "assistant"
            return {"content": content, "role": role}

    return None


def _extract_record_message(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize message extraction across Claude and Codex record variants."""
    message = record.get("message")
    if isinstance(message, dict):
        return cast(dict[str, Any], message)

    for key in ("payload", "metadata", "meta", "session_meta", "source"):
        payload = _to_dict_payload(record.get(key))
        if payload is None:
            continue
        message_candidate = _extract_message_like_block(payload)
        if isinstance(message_candidate, dict):
            return message_candidate

    content = record.get("content")
    if content is not None:
        role = record.get("role", record.get("type", "assistant"))
        if not isinstance(role, str):
            role = "assistant"
        return {"content": content, "role": role}

    if message is not None:
        role = record.get("role", record.get("type", "assistant"))
        if not isinstance(role, str):
            role = "assistant"
        return {"content": message, "role": role}

    return {}


def _extract_session_title(record: dict[str, Any]) -> str | None:
    """Extract a human-readable thread/session name from a record."""
    candidate_keys = (
        "thread_name",
        "name",
        "title",
        "session_name",
        "conversation_name",
    )

    for key in candidate_keys:
        value = _coerce_str(record.get(key))
        if value:
            return value

    for container_key in ("payload", "metadata", "meta", "session_meta"):
        bucket = _to_dict_payload(record.get(container_key))
        if bucket is None:
            continue
        for key in candidate_keys:
            value = _coerce_str(bucket.get(key))
            if value:
                return value

    return None


def _resolve_codex_index_root(codex_root: Path) -> Path | None:
    """Find the nearest ancestor containing a session_index.jsonl file."""
    current = codex_root
    for candidate in [current, *current.parents]:
        if (candidate / "session_index.jsonl").exists():
            return candidate
    return None


def _extract_session_id(record: dict) -> str | None:
    """Extract a stable session/thread id from a normalized record."""
    for key in ("thread_id", "threadId", "session_id", "uuid", "id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for container in ("payload", "metadata", "meta", "session_meta"):
        bucket = record.get(container)
        if isinstance(bucket, dict):
            for key in ("thread_id", "threadId", "session_id", "uuid", "id"):
                value = bucket.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _load_codex_session_names(codex_home: Path) -> dict[str, str]:
    """Load latest thread-name overrides from Codex session index."""
    index_path = codex_home / "session_index.jsonl"
    names: dict[str, str] = {}
    if not index_path.exists():
        return names

    try:
        with open(index_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(entry, dict):
                    continue
                session_id = _coerce_str(entry.get("id"))
                if session_id is None:
                    session_id = _coerce_str(entry.get("uuid"))
                if not session_id:
                    continue
                thread_name = None
                for key in ("thread_name", "name", "title", "session_name"):
                    candidate = _coerce_str(entry.get(key))
                    if candidate:
                        thread_name = candidate
                        break
                if thread_name:
                    names[session_id] = thread_name
    except (OSError, PermissionError):
        return {}

    return names


def _derive_codex_project_name(parent_dir: Path, root: Path) -> str | None:
    """Infer project name from a rollout directory under the sessions root."""
    if not parent_dir.is_relative_to(root):
        return parent_dir.name

    try:
        rel = parent_dir.relative_to(root)
    except ValueError:
        return parent_dir.name

    parts = rel.parts
    non_date_parts = [p for p in parts if not (p.isdigit() and len(p) in {2, 4})]
    if non_date_parts:
        return non_date_parts[-1]
    if parts:
        return parts[-1]
    return None


def _extract_session_id_from_file(path: Path) -> str | None:
    """Read a session file and return a best-effort session identifier."""
    try:
        with open(path, "r", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue

                record = _normalize_session_record(record)
                if not isinstance(record, dict):
                    continue

                source_id = _extract_session_id(record)
                if source_id:
                    return source_id

                # Fallback to message id for records that only identify in payload.
                message = _extract_record_message(record)
                message_id = _extract_session_id(message) if isinstance(message, dict) else None
                if message_id:
                    return message_id
    except (OSError, PermissionError):
        return None
    return None


def _extract_codex_session_id_from_filename(path: Path) -> str | None:
    """Extract a UUID-like session id from a Codex filename."""
    m = ROLLOUT_ID_RE.match(path.name.lower())
    if m:
        return m.group(1)
    m = UUID_IN_NAME_RE.match(path.name.lower())
    if m and UUID_RE.match(m.group(1)):
        return m.group(1)
    return None


def _extract_codex_bundle_id(
    path: Path,
    filename_session_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Extract per-file session id and thread-bundle id from a Codex session file.

    The bundle id follows the parent thread when available (subagent rollouts),
    so all trace files under the same logical session lineage are merged.
    """

    session_id: str | None = None
    bundle_id: str | None = None

    def _extract_parent_thread_id(bucket: dict[str, Any] | None) -> str | None:
        if bucket is None:
            return None

        # Common parent-thread shape for subagent run records.
        source = _to_dict_payload(bucket.get("source"))
        if source is None:
            return None
        subagent = _to_dict_payload(source.get("subagent"))
        if subagent is None:
            return None
        thread_spawn = _to_dict_payload(subagent.get("thread_spawn"))
        if thread_spawn is None:
            return None
        return _coerce_str(thread_spawn.get("parent_thread_id"))

    def _extract_first_id(bucket: dict[str, Any] | None) -> str | None:
        if bucket is None:
            return None

        for key in (
            "thread_id",
            "turn_id",
            "session_id",
            "conversation_id",
            "thread",
            "uuid",
            "id",
        ):
            value = _coerce_str(bucket.get(key))
            if value:
                return value

        nested = _to_dict_payload(bucket.get("source"))
        if nested is not None:
            nested_id = _extract_first_id(nested)
            if nested_id:
                return nested_id

        nested = _to_dict_payload(bucket.get("turn_context"))
        if nested is not None:
            nested_id = _extract_first_id(nested)
            if nested_id:
                return nested_id

        payload = _to_dict_payload(bucket.get("payload"))
        if payload is not None:
            payload_id = _extract_first_id(payload)
            if payload_id:
                return payload_id

        return None

    def _extract_nested_id(bucket: dict[str, Any] | None) -> str | None:
        if bucket is None:
            return None
        for value in bucket.values():
            if isinstance(value, dict):
                nested = _extract_first_id(value)
                if nested:
                    return nested
                nested = _extract_nested_id(value)
                if nested:
                    return nested
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        nested = _extract_first_id(item)
                        if nested:
                            return nested
                        nested = _extract_nested_id(item)
                        if nested:
                            return nested
                    elif isinstance(item, str):
                        item_text = item.strip()
                        if UUID_RE.match(item_text):
                            return item_text
            else:
                item_text = _coerce_str(value)
                if item_text and UUID_RE.match(item_text):
                    return item_text
        return None

    try:
        with open(path, "r", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue

                record = _normalize_session_record(record)
                if not isinstance(record, dict):
                    continue
                record = cast(dict[str, Any], record)

                if session_id is None:
                    session_id = _extract_session_id(record)

                for container in ("payload", "metadata", "meta", "session_meta", "source"):
                    bucket = _to_dict_payload(record.get(container))
                    if bucket is None:
                        continue
                    if bundle_id is None:
                        bundle_id = _extract_parent_thread_id(bucket)
                    if bundle_id is not None:
                        break
                    container_session_id = _extract_first_id(bucket)
                    if container_session_id is None:
                        container_session_id = _extract_nested_id(bucket)
                    if container_session_id and session_id is None:
                        session_id = container_session_id

                if session_id is not None and bundle_id is not None:
                    break
    except (OSError, PermissionError):
        return None, None

    if session_id is None:
        session_id = filename_session_id
    if session_id is None:
        return None, None

    norm_session_id = session_id.lower() if UUID_RE.match(_coerce_str(session_id) or "") else session_id

    if bundle_id is None:
        bundle_id = session_id

    normalized_bundle = bundle_id.lower() if UUID_RE.match(_coerce_str(bundle_id) or "") else bundle_id
    return norm_session_id, normalized_bundle


def _extract_codex_bundle_metadata(
    paths: list[Path],
    *,
    allow_heuristic_branch: bool = True,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    datetime | None,
    datetime | None,
    Path | None,
    Path | None,
]:
    """Extract best-effort metadata across multiple Codex files in one bundle."""
    discovered_project: str | None = None
    discovered_session_title: str | None = None
    discovered_source_id: str | None = None
    discovered_cwd: str | None = None
    branch_last_ts: datetime | None = None
    last_ts: datetime | None = None
    source_path: Path | None = None
    source_root: Path | None = None

    branch: str | None = None

    for p in sorted(paths, key=lambda pth: pth.stat().st_mtime, reverse=True):
        (
            p_branch,
            p_branch_last_ts,
            p_project,
            p_source_id,
            p_cwd,
            p_source_path,
            p_source_root,
            p_session_title,
        ) = _extract_branch_metadata(
            p,
            allow_heuristic_branch=allow_heuristic_branch,
        )

        if discovered_project is None and p_project is not None:
            discovered_project = p_project
            if discovered_session_title is None and p_session_title is not None:
                discovered_session_title = p_session_title
        if discovered_source_id is None and p_source_id is not None:
            discovered_source_id = p_source_id
        if discovered_cwd is None and p_cwd is not None:
            discovered_cwd = p_cwd

        if p_source_path is not None and source_path is None:
            source_path = p_source_path
        if p_source_root is not None and source_root is None:
            source_root = p_source_root

        if p_branch_last_ts is not None:
            aware_branch_ts = _ensure_aware_utc(p_branch_last_ts)
            if aware_branch_ts is not None and (
                branch_last_ts is None or aware_branch_ts > branch_last_ts
            ):
                branch_last_ts = aware_branch_ts
                branch = p_branch

        if p_branch is None:
            continue
        if last_ts is None:
            if p_branch_last_ts is not None:
                last_ts = _ensure_aware_utc(p_branch_last_ts)
            else:
                last_ts = None
            if branch is None:
                branch = p_branch
        else:
            if p_branch_last_ts is None:
                continue
            aware_current = _ensure_aware_utc(p_branch_last_ts)
            aware_last = _ensure_aware_utc(last_ts)
            if aware_last is None or aware_current is not None and aware_current > aware_last:
                last_ts = aware_current
                if branch is None:
                    branch = p_branch

    if branch is not None:
        normalized = branch.strip()
        if normalized.lower() in {
            "main",
            "master",
            "default",
            "cli",
            "rollout",
            "unresolved",
        }:
            branch = None

    return (
        discovered_project,
        discovered_session_title,
        discovered_source_id,
        discovered_cwd,
        branch,
        branch_last_ts,
        last_ts,
        source_path,
        source_root,
    )


def _extract_text_from_content(content) -> str:
    """Extract plain text from message content (string or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == BlockType.TEXT:
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def _render_tool_use(block: dict) -> Exchange:
    """Render a tool_use block as a one-liner Exchange."""
    name = block.get("name", "unknown")
    inp = block.get("input", {})

    if name == "Bash":
        cmd = inp.get("command", "")
        # Truncate long commands
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
        desc = cmd
    elif name == "Read":
        desc = inp.get("file_path", inp.get("path", ""))
    elif name == "Write":
        desc = inp.get("file_path", inp.get("path", ""))
    elif name == "Edit":
        desc = inp.get("file_path", inp.get("path", ""))
    elif name == "Glob":
        desc = inp.get("pattern", "")
    elif name == "Grep":
        desc = inp.get("pattern", "")
    elif name == "Agent":
        desc = inp.get("prompt", inp.get("description", ""))
        if len(desc) > 80:
            desc = desc[:77] + "..."
    else:
        # Generic: show first string value
        for v in inp.values():
            if isinstance(v, str) and v:
                desc = v[:80]
                break
        else:
            desc = ""

    return Exchange(
        role="tool",
        text=f"[{name}: {desc}]",
        tool_name=name,
    )


def _render_tool_result(block: dict) -> Exchange:
    """Render a tool_result block as a one-liner Exchange."""
    content = block.get("content", "")
    is_error = block.get("is_error", False)
    if isinstance(content, str):
        lines = content.strip().split("\n")
        line_count = len(lines)
        if is_error:
            preview = lines[0][:80] if lines else "error"
            status = "error"
        else:
            status = "ok" if line_count <= 3 else f"{line_count} lines"
            preview = lines[0][:80] if lines else ""
    elif isinstance(content, list):
        # Content blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == BlockType.TEXT:
                text_parts.append(item.get("text", ""))
        combined = "\n".join(text_parts)
        lines = combined.strip().split("\n")
        line_count = len(lines)
        status = "ok" if line_count <= 3 else f"{line_count} lines"
        preview = lines[0][:80] if lines else ""
        is_error = False
    else:
        status = "ok"
        preview = ""

    return Exchange(
        role="tool",
        text=f"-> {status}" + (f": {preview}" if preview else ""),
        tool_status=status,
        is_error=is_error,
    )


def parse_tail(
    path: Path, tail_bytes: int = 50 * 1024
) -> tuple[list[Exchange], int, datetime | None]:
    """Read the last tail_bytes of a session file and extract metadata.

    Adaptively expands the read window if the tail is all metadata noise
    (progress messages can be 5KB+ each, filling a small tail window).

    Returns:
        (exchanges, token_estimate, last_timestamp)
    """
    try:
        file_size = path.stat().st_size
    except (OSError, PermissionError):
        return [], 0, None

    if file_size == 0:
        return [], 0, None

    # Try increasing tail sizes until we find content or exhaust the file
    for try_bytes in [tail_bytes, 200 * 1024, 500 * 1024, file_size]:
        read_bytes = min(try_bytes, file_size)
        seeked = False
        try:
            with open(path, "r", errors="replace") as f:
                if file_size > read_bytes:
                    f.seek(file_size - read_bytes)
                    seeked = True
                raw_lines = f.readlines()
        except (OSError, PermissionError):
            return [], 0, None

        if seeked and raw_lines:
            raw_lines = raw_lines[1:]

        # Pass 0 as fallback_file_size so heuristic doesn't trigger —
        # we only want to stop expanding if we found real content or usage.
        exchanges, token_est, last_ts = _parse_raw_lines(raw_lines, 0)
        if exchanges:
            # Found content. If we also found usage, great. If not,
            # fall back to heuristic now.
            if token_est == 0:
                token_est = file_size // 14
            return exchanges, token_est, last_ts
        if read_bytes >= file_size:
            break  # read the whole file, nothing more to try

    # Final attempt with heuristic fallback
    return _parse_raw_lines(raw_lines, file_size)


def parse_tail_from_content(
    content: str, file_size: int = 0
) -> tuple[list[Exchange], int, datetime | None]:
    """Parse exchanges from raw string content (e.g., from git show).

    Same logic as parse_tail but takes content directly instead of reading a file.
    file_size is used for heuristic token estimate fallback if no usage data found.
    """
    if not content:
        return [], 0, None
    raw_lines = content.split("\n")
    return _parse_raw_lines(raw_lines, file_size)


def _parse_raw_lines(
    raw_lines: list[str], fallback_file_size: int = 0
) -> tuple[list[Exchange], int, datetime | None]:
    """Core parsing logic for session JSONL lines."""
    exchanges: list[Exchange] = []
    token_estimate = 0
    last_timestamp: datetime | None = None
    found_usage = False

    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        record = _normalize_session_record(record)
        if not isinstance(record, dict):
            continue
        normalized_record = cast(dict[str, Any], record)

        # Track timestamp
        ts = _ensure_aware_utc(_parse_timestamp(_coerce_str(normalized_record.get("timestamp"))))
        if ts is not None:
            last_timestamp = ts

        rtype = _coerce_str(normalized_record.get("type")) or ""

        # Skip noise types
        if rtype in SKIP_TYPES:
            continue

        msg = _extract_record_message(normalized_record)
        if not msg:
            continue
        msg_role = msg.get("role")
        role_hint = str(msg_role).lower() if isinstance(msg_role, str) else ""
        role_hint = role_hint if role_hint in ("user", "assistant", "system", "developer", "tool") else ""

        content = msg.get("content")

        # Extract usage for token estimate
        usage = msg.get("usage")
        if isinstance(usage, dict):
            found_usage = True
            token_estimate = (
                usage.get("input_tokens", 0)
                + usage.get("cache_read_input_tokens", 0)
                + usage.get("cache_creation_input_tokens", 0)
            )

        # Skip thinking blocks in content
        if isinstance(content, list):
            # Check for tool_use (assistant calling tools)
            for block in content:
                if not isinstance(block, dict):
                    if isinstance(block, str):
                        text = block.strip()
                        if text:
                            role = role_hint if role_hint in ("user", "assistant") else (
                                "assistant" if rtype == "assistant" else "user"
                            )
                            exchanges.append(Exchange(role=role, text=text))
                    continue
                btype = block.get("type", "")
                if btype == "thinking":
                    continue
                if btype == "tool_use":
                    exchanges.append(_render_tool_use(block))
                elif btype == "tool_result":
                    exchanges.append(_render_tool_result(block))
                elif btype == "text":
                    text = block.get("text", "").strip()
                    if text:
                        role = role_hint if role_hint in ("user", "assistant") else (
                            "assistant" if rtype == "assistant" else "user"
                        )
                        exchanges.append(Exchange(role=role, text=text))
                elif btype in {"output_text", "input_text", "assistant_text", "text_block"}:
                    text = block.get("text", "").strip()
                    if text:
                        role = role_hint if role_hint in ("user", "assistant") else (
                            "assistant" if rtype == "assistant" else "user"
                        )
                        exchanges.append(Exchange(role=role, text=text))
                elif "text" in block:
                    text = str(block.get("text", "")).strip()
                    if text:
                        role = role_hint if role_hint in ("user", "assistant") else (
                            "assistant" if rtype == "assistant" else "user"
                        )
                        exchanges.append(Exchange(role=role, text=text))
        elif isinstance(content, str) and content.strip():
            if role_hint in ("user", "assistant"):
                role = role_hint
            else:
                role = rtype if rtype in ("user", "assistant") else "user"
            exchanges.append(Exchange(role=role, text=content.strip()))

    # Fallback token estimate
    if not found_usage and fallback_file_size > 0:
        token_estimate = fallback_file_size // 14

    return exchanges, token_estimate, last_timestamp


def _extract_branch_metadata(
    path: Path,
    *,
    allow_heuristic_branch: bool = True,
) -> tuple[
    str | None,
    datetime | None,
    str | None,
    str | None,
    str | None,
    Path | None,
    Path | None,
    str | None,
]:
    """Extract branch name and max branch-related timestamp from a session file."""

    def _coerce_str(value: object) -> str | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text if text else None

    def _extract_from_mapping(mapping: dict[str, Any], *keys: str) -> str | None:
        for key in keys:
            candidate = _coerce_str(mapping.get(key))
            if candidate:
                return candidate
        return None

    def _extract_source_branch(source: object) -> str | None:
        if isinstance(source, dict):
            if allow_heuristic_branch:
                keys = ("branch", "gitBranch", "git_branch")
            else:
                keys = ("gitBranch", "git_branch")
            candidate = _extract_from_mapping(cast(dict[str, Any], source), *keys)
            return candidate
        if isinstance(source, str):
            return _coerce_str(source)
        return None

    def _branch_candidate_is_plausible(candidate: str) -> bool:
        if not candidate:
            return False
        if "/" in candidate and candidate.count("/") > 2:
            return False
        if " " in candidate and len(candidate.split()) > 3:
            return False
        forbidden = {
            "cli",
            "main",
            "rollout",
            "master",
            "current",
            "unified_exec_startup",
            "exec_command",
            "reduced",
            "reduced_session",
            "unknown",
            "None",
            "null",
        }
        if candidate in forbidden:
            return False
        if candidate.startswith(("main/", "refs/heads/")):
            return False
        return True

    def _derive_branch_from_head(cwd: str) -> str | None:
        git_path = Path(cwd) / ".git"
        if not git_path.exists():
            return None

        try:
            if git_path.is_file():
                target = git_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not target.startswith("gitdir:"):
                    return None
                git_dir = Path(target.removeprefix("gitdir:").strip())
                if not git_dir.is_absolute():
                    git_dir = (git_path.parent / git_dir).resolve()
                git_path = git_dir
            head = (git_path / "HEAD").read_text(
                encoding="utf-8", errors="ignore"
            ).strip()
        except OSError:
            return None

        if not head.startswith("ref:"):
            return None
        ref = head.removeprefix("ref:").strip()
        if not ref:
            return None
        return ref.rsplit("/", 1)[-1]

    def _derive_project_name(payload: dict[str, Any]) -> str | None:
        cwd = _coerce_str(payload.get("cwd"))
        if not cwd:
            return None
        p = Path(cwd)
        if not p.is_absolute():
            return None
        parts = [part for part in p.parts if part not in {"", "/", "Users", "home"}]
        return parts[-1] if parts else None

    def _extract_from_record_paths(record: dict[str, Any]) -> tuple[str | None, str | None]:
        for key in (
            "payload",
            "session_meta",
            "metadata",
            "meta",
            "source",
        ):
            bucket = _to_dict_payload(record.get(key))
            if bucket is None:
                continue
            project = _derive_project_name(bucket)
            if project is not None or "cwd" in bucket:
                return project, _coerce_str(bucket.get("cwd"))
        return None, _coerce_str(record.get("cwd"))

    branch: str | None = None
    branch_last_timestamp: datetime | None = None
    project_name: str | None = None
    cwd: str | None = None
    source_id: str | None = None
    session_title: str | None = None
    source_path: Path | None = None
    source_root: Path | None = None

    try:
        with open(path, "r", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue

                record = _normalize_session_record(record)
                if not isinstance(record, dict):
                    continue
                record = cast(dict[str, Any], record)

                ts = _ensure_aware_utc(_parse_timestamp(_coerce_str(record.get("timestamp"))))
                if ts is not None:
                    prior = _ensure_aware_utc(branch_last_timestamp)
                    if prior is None or ts > prior:
                        branch_last_timestamp = ts

                if project_name is None:
                    discovered_project, discovered_cwd = _extract_from_record_paths(record)
                    project_name = discovered_project
                    cwd = discovered_cwd

                if allow_heuristic_branch:
                    branch_candidate = _extract_from_mapping(record, "gitBranch", "branch", "git_branch")
                    if not branch_candidate:
                        msg = _extract_record_message(record)
                        if msg:
                            branch_candidate = _extract_from_mapping(
                                msg, "gitBranch", "branch", "git_branch"
                            )
                    if not branch_candidate:
                        metadata = _to_dict_payload(record.get("metadata"))
                        if metadata is not None:
                            branch_candidate = _extract_from_mapping(
                                metadata, "gitBranch", "branch", "git_branch"
                            )
                            if not branch_candidate:
                                payload = _to_dict_payload(record.get("payload"))
                                if payload is not None:
                                    branch_candidate = _extract_from_mapping(
                                        payload,
                                        "gitBranch",
                                        "branch",
                                        "git_branch",
                                    )
                                    if not branch_candidate:
                                        branch_candidate = _extract_source_branch(
                                            payload.get("source")
                                        )
                    if not branch_candidate:
                        branch_candidate = _extract_source_branch(record.get("source"))
                        if branch_candidate == "main":
                            branch_candidate = None

                    if not branch_candidate and cwd:
                        branch_candidate = _derive_branch_from_head(cwd)

                    if branch_candidate and _branch_candidate_is_plausible(branch_candidate):
                        branch = branch_candidate

                if session_title is None:
                    session_title = _extract_session_title(record)

                if source_id is None:
                    for key in (
                        "thread_id",
                        "turn_id",
                        "threadId",
                        "session_id",
                        "conversation_id",
                        "uuid",
                        "id",
                    ):
                        candidate = _coerce_str(record.get(key))
                        if candidate:
                            source_id = candidate
                    for key in ("payload", "metadata", "meta", "session_meta"):
                        bucket = _to_dict_payload(record.get(key))
                        if bucket is None:
                            continue
                        source_id = _extract_from_mapping(
                            bucket,
                            "thread_id",
                            "turn_id",
                            "threadId",
                            "session_id",
                            "uuid",
                            "id",
                        )
                        if source_id:
                            break

                if source_root is None:
                    discovered_root = _coerce_str(record.get("source_root"))
                    if discovered_root:
                        source_root = Path(discovered_root)
                    source_path_candidate = _coerce_str(record.get("source_path"))
                    if source_path_candidate:
                        source_path = Path(source_path_candidate)

        if allow_heuristic_branch and branch is None and cwd:
            branch = _derive_branch_from_head(cwd)

        if not source_id:
            source_id = path.stem

        if source_root is None and cwd:
            source_root = Path(cwd)

        if source_path is None:
            source_path = path

        return (
            branch,
            branch_last_timestamp,
            project_name,
            source_id,
            cwd,
            source_path,
            source_root,
            session_title,
        )
    except (OSError, PermissionError):
        return None, None, None, None, None, None, None, None


def _count_lines(path: Path) -> int:
    """Count newlines in file without parsing JSON."""
    try:
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except (OSError, PermissionError):
        return 0


def compute_density_profile(
    path: Path, buckets: int = 40, tail_bytes: int = 200 * 1024
) -> list[int]:
    """Compute content chars per positional bucket from session file.

    Adaptively expands the read window (like parse_tail) to find content
    through progress noise. Returns a list of `buckets` integers.
    """
    profile = [0] * buckets

    try:
        file_size = path.stat().st_size
    except (OSError, PermissionError):
        return profile

    if file_size == 0:
        return profile

    # Adaptive: expand window until we find content lines
    raw_lines = []
    for try_bytes in [tail_bytes, 500 * 1024, file_size]:
        read_bytes = min(try_bytes, file_size)
        seeked = False
        try:
            with open(path, "r", errors="replace") as f:
                if file_size > read_bytes:
                    f.seek(file_size - read_bytes)
                    seeked = True
                raw_lines = f.readlines()
        except (OSError, PermissionError):
            return profile

        if seeked and raw_lines:
            raw_lines = raw_lines[1:]

        # Check if we found any content (not just noise)
        has_content = False
        for raw in raw_lines:
            try:
                record = json.loads(raw.strip())
                if record.get("type") in ("user", "assistant"):
                    has_content = True
                    break
            except (json.JSONDecodeError, ValueError):
                continue
        if has_content or read_bytes >= file_size:
            break

    # Filter out noise lines and collect content chars per line
    content_lines: list[int] = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        record = _normalize_session_record(record)
        if not isinstance(record, dict):
            continue

        record = cast(dict[str, Any], record)
        rtype = _coerce_str(record.get("type")) or ""
        if rtype in SKIP_TYPES:
            continue

        # Sum content chars from this record
        msg = record.get("message", {})
        if not isinstance(msg, dict):
            content_lines.append(0)
            continue

        content = cast(dict[str, Any], msg).get("content")
        chars = 0
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for key in ("text", "thinking", "content"):
                        val = block.get(key, "")
                        if isinstance(val, str):
                            chars += len(val)
        content_lines.append(chars)

    if not content_lines:
        return profile

    total_lines = len(content_lines)
    for i, chars in enumerate(content_lines):
        bucket = min(int(i / total_lines * buckets), buckets - 1)
        profile[bucket] += chars

    return profile


def scan_projects(projects_dir: Path, provider: str = "claude") -> list[SessionInfo]:
    """Scan all subdirectories for session JSONL files.

    Returns SessionInfo list sorted by project name (alphabetical),
    then newest first within each project.
    """
    if not projects_dir.is_dir():
        return []

    if provider != "codex":
        project_dirs = sorted(projects_dir.iterdir())
    else:
        project_dirs = [projects_dir]

    sessions: list[SessionInfo] = []

    for proj_dir in project_dirs:
        if not proj_dir.is_dir():
            continue

        try:
            if provider == "codex":
                index_root = _resolve_codex_index_root(proj_dir)
                codex_names = _load_codex_session_names(index_root or proj_dir.parent)
                files = list(proj_dir.rglob("*.jsonl"))
            else:
                files = list(proj_dir.iterdir())
        except PermissionError:
            continue

        grouped_bundle_files: dict[tuple[str, str], list[Path]] = {}
        main_files: dict[tuple[Path, str], list[Path]] = {}
        # Use a bundle key (root thread) rather than raw session id so
        # subagent rollouts stay in the parent lineage.
        # Use bundle key globally, not per date-folder, so one session lineage is
        # shown as a single entry across date-partitioned folders.

        if provider == "codex":
            for f in files:
                if not f.name.endswith(".jsonl") or any(
                    f.name.endswith(sfx) for sfx in SKIP_SUFFIXES
                ):
                    continue

                filename_session_id = _extract_codex_session_id_from_filename(f)
                session_id, bundle_id = _extract_codex_bundle_id(
                    f, filename_session_id=filename_session_id
                )
                if session_id is None:
                    continue
                if bundle_id is None:
                    bundle_id = session_id

                (
                    _,
                    _,
                    project_name,
                    _,
                    _,
                    source_path,
                    source_root,
                    discovered_session_title,
                ) = _extract_branch_metadata(f, allow_heuristic_branch=True)

                if project_name is None:
                    if source_root is not None:
                        project_name = _derive_codex_project_name(source_root, source_root)
                    if project_name is None:
                        project_name = "codex"

                group_key = (bundle_id, project_name)
                grouped_bundle_files.setdefault(group_key, []).append(f)
                if discovered_session_title and session_id:
                    codex_names.setdefault(session_id, discovered_session_title)

            # Codex rollout continuations: a ``rollout-*.jsonl`` file with no
            # matching non-rollout sibling in the same bundle should be merged
            # into the directory's primary bundle if one exists. The lineage
            # marker comes from filesystem locality, not record IDs, because
            # rollout files don't always carry the parent thread's id.
            def _is_rollout(path: Path) -> bool:
                return ROLLOUT_RE.match(path.name) is not None

            primary_by_dir: dict[Path, tuple[str | None, str]] = {}
            for (b_id, p_name), paths in grouped_bundle_files.items():
                if any(not _is_rollout(p) for p in paths):
                    for p in paths:
                        if not _is_rollout(p):
                            primary_by_dir.setdefault(p.parent, (b_id, p_name))

            merged_groups: dict[tuple[str | None, str], list[Path]] = {}
            for (b_id, p_name), paths in grouped_bundle_files.items():
                if all(_is_rollout(p) for p in paths):
                    # Rollout-only bundle: try to fold into directory primary.
                    parents = {p.parent for p in paths}
                    if len(parents) == 1:
                        primary_key = primary_by_dir.get(next(iter(parents)))
                        if primary_key is not None:
                            merged_groups.setdefault(primary_key, []).extend(paths)
                            continue
                merged_groups.setdefault((b_id, p_name), []).extend(paths)
            grouped_bundle_files = cast(
                dict[tuple[str, str], list[Path]], merged_groups
            )

            for (bundle_id, project_name), paths in grouped_bundle_files.items():
                if not paths:
                    continue
                all_paths = sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)
                if not all_paths:
                    continue

                # Derive metadata from all files in the bundle.
                (
                    discovered_project,
                    discovered_session_title,
                    discovered_source_id,
                    _,
                    discovered_branch,
                    discovered_branch_last_ts,
                    _,
                    discovered_source_path,
                    _,
                ) = _extract_codex_bundle_metadata(
                    all_paths, allow_heuristic_branch=True
                )

                if discovered_branch:
                    normalized = discovered_branch.strip()
                    if normalized.lower() in {"main", "master", "default", "cli", "rollout"}:
                        discovered_branch = None

                branch_last_ts = discovered_branch_last_ts
                branch = discovered_branch
                project_name = discovered_project
                project_slug = ""
                resolved = None

                if project_name is None:
                    latest_path = sorted(all_paths, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                    project_name = _derive_codex_project_name(latest_path.parent, proj_dir)
                    if project_name is None or project_name.isdigit():
                        project_name = "codex"
                    project_slug = project_name
                else:
                    project_slug = project_name

                # Choose the first non-rollout file (latest) as the replay anchor.
                non_rollout_paths = [p for p in all_paths if not ROLLOUT_RE.match(p.name)]
                session_display_path: Path | None = None
                best_display_score = (-1, -1, -1.0)
                ranked_paths = sorted(all_paths, key=lambda p: p.stat().st_mtime, reverse=True)
                for candidate in ranked_paths:
                    if non_rollout_paths and ROLLOUT_RE.match(candidate.name):
                        continue
                    try:
                        exchanges, tokens, _ = parse_tail(candidate)
                    except Exception:
                        exchanges = []
                        tokens = 0
                    candidate_score = (
                        1 if exchanges else 0,
                        1 if tokens > 0 else 0,
                        candidate.stat().st_mtime,
                    )
                    if candidate_score > best_display_score:
                        session_display_path = candidate
                        best_display_score = candidate_score
                if session_display_path is None:
                    session_display_path = ranked_paths[0]

                bundle_path_session_id = bundle_id or discovered_source_id or session_display_path.stem
                if discovered_session_title is None and bundle_path_session_id:
                    discovered_session_title = codex_names.get(bundle_path_session_id, None)

                parse_error = False
                last_exchanges: list[Exchange] = []
                token_estimate = 0
                last_ts = None
                size = 0
                line_count = 0
                try:
                    for p in all_paths:
                        size += p.stat().st_size
                        line_count += _count_lines(p)
                        ex, tok, ts = parse_tail(p)
                        if ex:
                            last_exchanges.extend(ex)
                        if tok > 0:
                            token_estimate = max(token_estimate, tok)
                        ts_aware = _ensure_aware_utc(ts)
                        if ts_aware is not None:
                            last_ts_aware = _ensure_aware_utc(last_ts)
                            if last_ts is None or last_ts_aware is None or ts_aware > last_ts_aware:
                                last_ts = ts_aware
                except Exception:
                    last_exchanges = []
                    last_ts = None
                    token_estimate = 0
                    parse_error = True

                if size == 0:
                    continue

                if token_estimate == 0:
                    token_estimate = size // 14

                if branch_last_ts is None:
                    branch_last_ts = last_ts

                age_display = format_age(last_ts) if last_ts else "?"
                try:
                    density = compute_density_profile(session_display_path)
                except Exception:
                    density = []

                sessions.append(
                    SessionInfo(
                        path=session_display_path,
                        project_name=project_name,
                        session_id=bundle_path_session_id,
                        short_id=(bundle_path_session_id[:8] if bundle_path_session_id else session_display_path.stem[:8]),
                        size_bytes=size,
                        token_estimate=token_estimate,
                        last_timestamp=last_ts,
                        age_display=age_display,
                        line_count=line_count,
                        continuation_files=[p for p in all_paths if p != session_display_path],
                        last_exchanges=last_exchanges,
                        parse_error=parse_error,
                        density_profile=density,
                        resolved_dir=resolved,
                        is_dangling=False,
                        project_slug=project_slug,
                        provider=provider,
                        # Codex sessions without an explicit branch fall back
                        # to "main" so the UI always has a concrete label.
                        branch=branch or "main",
                        branch_last_timestamp=branch_last_ts,
                        source_format="codex",
                        source_id=bundle_path_session_id,
                        source_path=discovered_source_path or session_display_path,
                        session_title=discovered_session_title,
                    )
                )

            continue

        # Existing Claude/legacy scanning behavior.
        continuation_map: dict[tuple[Path, str], list[Path]] = {}
        for f in files:
            if not f.name.endswith(".jsonl") or any(f.name.endswith(sfx) for sfx in SKIP_SUFFIXES):
                continue
            stem = f.stem
            cont_match = CONTINUATION_RE.match(f.name)
            if cont_match:
                continuation_map.setdefault((proj_dir, cont_match.group(1)), []).append(f)
                continue
            ts_cont_match = TIMESTAMP_CONTINUATION_RE.match(f.name)
            if ts_cont_match and UUID_RE.match(ts_cont_match.group(1)):
                continuation_map.setdefault((proj_dir, ts_cont_match.group(1)), []).append(f)
                continue
            if UUID_RE.match(stem):
                main_files[(proj_dir, stem)] = [f]

        for (parent_dir, uuid), paths in main_files.items():
            path = paths[0] if paths else None
            if path is None:
                continue
            try:
                stat = path.stat()
            except (OSError, PermissionError):
                continue

            size = stat.st_size
            if size == 0:
                continue

            project_name = derive_project_name(parent_dir.name)
            resolved = get_resolved_path(parent_dir.name)
            project_slug = parent_dir.name
            line_count = _count_lines(path)
            parse_error = False
            last_exchanges: list[Exchange] = []
            token_estimate = size // 14
            last_ts = None
            try:
                ex, tok, last_ts = parse_tail(path)
                if ex:
                    last_exchanges.extend(ex)
                if tok:
                    token_estimate = tok
                if not last_exchanges:
                    token_estimate = sum(_count_lines(path) for _ in [path]) // 14

                for continuation_path in continuation_map.get((parent_dir, uuid), []):
                    ce, ct, _ = parse_tail(continuation_path)
                    if ce:
                        last_exchanges.extend(ce)
                    line_count += _count_lines(continuation_path)
                    if ct > 0:
                        token_estimate += ct
            except Exception:
                last_exchanges = []
                token_estimate = size // 14
                last_ts = None
                parse_error = True

            age_display = format_age(last_ts) if last_ts else "?"
            try:
                density = compute_density_profile(path)
            except Exception:
                density = []

            sessions.append(
                SessionInfo(
                    path=path,
                    project_name=project_name,
                    session_id=uuid,
                    short_id=uuid[:8],
                    size_bytes=size,
                    token_estimate=token_estimate,
                    last_timestamp=last_ts,
                    age_display=age_display,
                    line_count=line_count,
                    continuation_files=continuation_map.get((parent_dir, uuid), []),
                    last_exchanges=last_exchanges,
                    parse_error=parse_error,
                    density_profile=density,
                    resolved_dir=resolved,
                    is_dangling=resolved is None,
                    project_slug=project_slug,
                    provider=provider,
                    branch=project_name,
                    branch_last_timestamp=last_ts,
                )
            )

    # Sort: project name alphabetical, then newest first within project
    def sort_key(s: SessionInfo) -> tuple:
        # Use timestamp for ordering; None sorts last (oldest)
        ts = s.last_timestamp
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Negate timestamp for newest-first; use epoch 0 for None
        ts_val = ts.timestamp() if ts else 0.0
        return (s.project_name.lower(), -ts_val)

    sessions.sort(key=sort_key)
    return sessions
