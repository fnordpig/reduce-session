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

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
CONTINUATION_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.\d+.*\.jsonl$"
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


def derive_project_name(slug: str) -> str:
    """Convert directory slug to readable name.

    '-Users-rwaugh-src-mine-ripvec' -> 'ripvec'
    Uses last non-empty path component.
    """
    parts = [p for p in slug.split("-") if p]
    return parts[-1] if parts else slug


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


def _extract_text_from_content(content) -> str:
    """Extract plain text from message content (string or content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
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
            if isinstance(item, dict) and item.get("type") == "text":
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

        if not isinstance(record, dict):
            continue

        # Track timestamp
        ts = _parse_timestamp(record.get("timestamp"))
        if ts is not None:
            last_timestamp = ts

        rtype = record.get("type", "")

        # Skip noise types
        if rtype in SKIP_TYPES:
            continue

        msg = record.get("message", {})
        if not isinstance(msg, dict):
            continue

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
                        role = "assistant" if rtype == "assistant" else "user"
                        exchanges.append(Exchange(role=role, text=text))
        elif isinstance(content, str) and content.strip():
            role = msg.get("role", rtype)
            if role not in ("user", "assistant"):
                role = rtype if rtype in ("user", "assistant") else "user"
            exchanges.append(Exchange(role=role, text=content.strip()))

    # Fallback token estimate
    if not found_usage and fallback_file_size > 0:
        token_estimate = fallback_file_size // 14

    return exchanges, token_estimate, last_timestamp


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
    """Compute content chars per positional bucket from session tail.

    Returns a list of `buckets` integers, each representing chars in that
    fraction of the file. Used for heatmap sparklines.
    """
    profile = [0] * buckets

    try:
        file_size = path.stat().st_size
    except (OSError, PermissionError):
        return profile

    if file_size == 0:
        return profile

    seeked = False
    try:
        with open(path, "r", errors="replace") as f:
            if file_size > tail_bytes:
                f.seek(file_size - tail_bytes)
                seeked = True
            raw_lines = f.readlines()
    except (OSError, PermissionError):
        return profile

    # Skip truncated first line if we seeked into the middle
    if seeked and raw_lines:
        raw_lines = raw_lines[1:]

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

        if not isinstance(record, dict):
            continue

        rtype = record.get("type", "")
        if rtype in SKIP_TYPES:
            continue

        # Sum content chars from this record
        msg = record.get("message", {})
        if not isinstance(msg, dict):
            content_lines.append(0)
            continue

        content = msg.get("content")
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


def scan_projects(projects_dir: Path) -> list[SessionInfo]:
    """Scan all subdirectories for session JSONL files.

    Returns SessionInfo list sorted by project name (alphabetical),
    then newest first within each project.
    """
    if not projects_dir.is_dir():
        return []

    sessions: list[SessionInfo] = []

    try:
        project_dirs = sorted(projects_dir.iterdir())
    except PermissionError:
        return []

    for proj_dir in project_dirs:
        if not proj_dir.is_dir():
            continue

        project_name = derive_project_name(proj_dir.name)

        try:
            files = list(proj_dir.iterdir())
        except PermissionError:
            continue

        # Separate main session files from continuations
        main_files: dict[str, Path] = {}  # uuid -> path
        continuation_map: dict[str, list[Path]] = {}  # uuid -> [cont paths]

        for f in files:
            fname = f.name

            # Skip non-jsonl and backup files
            if not fname.endswith(".jsonl"):
                continue
            if any(fname.endswith(sfx) for sfx in SKIP_SUFFIXES):
                continue

            # Check continuation pattern first
            cont_match = CONTINUATION_RE.match(fname)
            if cont_match:
                parent_uuid = cont_match.group(1)
                continuation_map.setdefault(parent_uuid, []).append(f)
                continue

            # Check main session file: UUID.jsonl
            stem = f.stem
            if UUID_RE.match(stem):
                main_files[stem] = f

        # Build SessionInfo for each main file
        for uuid, path in main_files.items():
            try:
                stat = path.stat()
            except (OSError, PermissionError):
                continue

            size = stat.st_size
            if size == 0:
                continue

            continuations = continuation_map.get(uuid, [])
            line_count = _count_lines(path)

            parse_error = False
            try:
                last_exchanges, token_estimate, last_ts = parse_tail(path)
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
                    continuation_files=continuations,
                    last_exchanges=last_exchanges,
                    parse_error=parse_error,
                    density_profile=density,
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
