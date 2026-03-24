# Session Browser TUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Consult context7 (`/websites/textual_textualize_io`) heavily for textual API details before writing any widget code.

**Goal:** Add a TUI for browsing Claude Code sessions across all projects, previewing conversation tails, and running reductions with a visual modal.

**Architecture:** Extract the reduction pipeline from cli.py into reduction.py, git operations into git_ops.py, add session discovery (session.py), then build the TUI (tui.py, widgets.py, styles.tcss) on top. The existing CLI stays functional — the TUI is an additional entry point.

**Tech Stack:** Python 3.10+, textual>=1.0, rich (transitive via textual)

**Spec:** `docs/superpowers/specs/2026-03-23-session-browser-tui-design.md`

---

## File Map

```
src/reduce_session/
    __init__.py          # unchanged
    cli.py               # MODIFY: slim down to arg parsing + dispatch, import from reduction/git_ops
    reduction.py         # CREATE: extraction from cli.py — pipeline, profiles, token budget, helpers
    git_ops.py           # CREATE: extraction from cli.py — all git preservation functions
    session.py           # CREATE: SessionInfo/Exchange dataclasses, scan_projects, parse_tail
    tui.py               # CREATE: SessionBrowserApp, screen composition, key handlers
    widgets.py           # CREATE: SessionTree, ConversationPreview, TokenGauge, ReduceModal
    styles.tcss          # CREATE: textual CSS for layout and theming
tests/
    __init__.py          # CREATE
    test_reduction.py    # CREATE: tests for the extracted reduction pipeline
    test_session.py      # CREATE: tests for session discovery and tail parsing
    test_git_ops.py      # CREATE: tests for git operations
    conftest.py          # CREATE: shared fixtures (temp dirs, sample JSONL files)
pyproject.toml           # MODIFY: add textual dependency, test dependency
```

---

### Task 1: Extract reduction.py from cli.py

The reduction pipeline is the core logic. Extract it first so both CLI and TUI can import it.

**Files:**
- Create: `src/reduce_session/reduction.py`
- Modify: `src/reduce_session/cli.py`
- Create: `tests/test_reduction.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create conftest.py with sample JSONL fixture**

```python
# tests/conftest.py
import json
import pytest
from pathlib import Path


@pytest.fixture
def sample_session(tmp_path):
    """Create a minimal but realistic session JSONL for testing."""
    messages = [
        {"type": "system", "uuid": "sys-1", "message": {"content": "You are Claude."}, "timestamp": "2026-03-23T01:00:00Z"},
        {"type": "user", "uuid": "u-1", "parentUuid": "sys-1", "message": {"content": "Hello"}, "timestamp": "2026-03-23T01:01:00Z"},
        {"type": "assistant", "uuid": "a-1", "parentUuid": "u-1", "message": {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}], "usage": {"input_tokens": 10, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 5}}, "timestamp": "2026-03-23T01:01:30Z"},
        {"type": "progress", "uuid": "p-1", "parentUuid": "a-1", "data": {"type": "hook_progress"}, "timestamp": "2026-03-23T01:01:31Z"},
        {"type": "user", "uuid": "u-2", "parentUuid": "p-1", "message": {"content": [{"type": "tool_result", "tool_use_id": "tu-1", "content": "file contents here " * 500}]}, "timestamp": "2026-03-23T01:02:00Z"},
        {"type": "assistant", "uuid": "a-2", "parentUuid": "u-2", "message": {"role": "assistant", "content": [{"type": "text", "text": "I see the file."}], "usage": {"input_tokens": 20, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 10}}, "timestamp": "2026-03-23T01:02:30Z"},
    ]
    path = tmp_path / "test-session.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


@pytest.fixture
def sample_project_dir(tmp_path, sample_session):
    """Create a project directory structure mimicking ~/.claude/projects/."""
    project = tmp_path / "projects" / "-Users-test-src-myproject"
    project.mkdir(parents=True)
    import shutil
    dest = project / "abc12345-dead-beef-cafe-123456789abc.jsonl"
    shutil.copy(sample_session, dest)
    return project
```

- [ ] **Step 2: Create tests/\_\_init\_\_.py**

Empty file.

- [ ] **Step 3: Write test for reduce_session() function**

```python
# tests/test_reduction.py
from reduce_session.reduction import reduce_session, ReductionResult


def test_reduce_session_returns_result(sample_session):
    result = reduce_session(str(sample_session))
    assert isinstance(result, ReductionResult)
    assert result.new_count <= result.orig_count
    assert result.new_size <= result.orig_size
    assert isinstance(result.stats, dict)


def test_reduce_session_strips_progress(sample_session):
    result = reduce_session(str(sample_session))
    assert result.stats.get("progress", 0) >= 1
    assert result.new_count < result.orig_count


def test_reduce_session_reparents_after_drop(sample_session):
    """Progress p-1 is parent of u-2. After dropping p-1, u-2 must point to a-1."""
    import json
    result = reduce_session(str(sample_session))
    for line in result.kept_lines:
        obj = json.loads(line)
        if obj.get("uuid") == "u-2":
            assert obj["parentUuid"] == "a-1", "u-2 should be reparented to a-1 after p-1 dropped"
            break


def test_reduce_session_strips_usage(sample_session):
    import json
    result = reduce_session(str(sample_session))
    for line in result.kept_lines:
        obj = json.loads(line)
        if obj.get("type") == "assistant":
            assert "usage" not in obj.get("message", {}), "usage should be stripped"


def test_reduce_session_with_token_estimate(sample_session):
    result = reduce_session(str(sample_session), estimate_tokens=True)
    assert result.orig_budget is not None
    assert result.reduced_budget is not None
    assert result.api_tokens is not None


def test_reduce_session_profiles(sample_session):
    gentle = reduce_session(str(sample_session), profile="gentle")
    aggressive = reduce_session(str(sample_session), profile="aggressive")
    assert aggressive.new_size <= gentle.new_size
```

- [ ] **Step 4: Run tests — expect failure (module doesn't exist yet)**

```bash
uv run pytest tests/test_reduction.py -v
```

Expected: `ModuleNotFoundError: No module named 'reduce_session.reduction'`

- [ ] **Step 5: Create reduction.py by extracting from cli.py**

Extract lines 34-164 (PROFILES, ENVELOPE_FIELDS, CHARS_PER_TOKEN) and lines 299-793 (aggressiveness functions through trim_toolUseResult), plus lines 325-484 (TokenBudget), plus lines 620-726 (detect_* functions), plus the `extract_last_usage` function. Add a new `reduce_session()` orchestrator and `ReductionResult` dataclass.

The orchestrator should contain the logic currently in `main()` lines 1143-1421 (passes 1-5), but return a `ReductionResult` instead of writing files.

```python
# src/reduce_session/reduction.py — top-level structure
"""Reduction pipeline for Claude Code session JSONL files."""

from __future__ import annotations
import hashlib
import json
import re
from dataclasses import dataclass, field

# ... PROFILES dict, ENVELOPE_FIELDS, CHARS_PER_TOKEN ...
# ... TokenBudget class ...
# ... all helper functions (truncate, trim_string, strip_*, detect_*, etc.) ...
# ... make_aggressiveness_fn, blended_limit ...

@dataclass
class ReductionResult:
    kept_lines: list[str]
    stats: dict[str, int]
    orig_count: int
    orig_size: int
    new_count: int
    new_size: int
    orig_budget: TokenBudget | None = None
    reduced_budget: TokenBudget | None = None
    api_tokens: int | None = None

def reduce_session(
    input_path: str,
    profile: str = "standard",
    cut: int = 50,
    fade: int = 75,
    estimate_tokens: bool = False,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> ReductionResult:
    """Run the full reduction pipeline. Returns results without writing."""
    # ... passes 1-5 extracted from cli.py main() ...
```

- [ ] **Step 6: Run tests — expect pass**

```bash
uv run pytest tests/test_reduction.py -v
```

- [ ] **Step 7: Update cli.py to import from reduction.py**

Replace the extracted code in cli.py with imports. `main()` calls `reduce_session()` then handles output formatting, `do_apply`, etc. Keep `parse_args`, `main`, and all the `do_*`/git functions in cli.py for now (git_ops extraction is next task).

- [ ] **Step 8: Verify CLI still works**

```bash
uv run reduce-session --dry-run --tokens ~/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl
```

Compare output to previous run — should be identical.

- [ ] **Step 9: Commit**

```bash
git add src/reduce_session/reduction.py src/reduce_session/cli.py tests/
git commit -m "refactor: extract reduction pipeline into reduction.py"
```

---

### Task 2: Extract git_ops.py from cli.py

**Files:**
- Create: `src/reduce_session/git_ops.py`
- Modify: `src/reduce_session/cli.py`
- Create: `tests/test_git_ops.py`

- [ ] **Step 1: Write tests for git operations**

```python
# tests/test_git_ops.py
import os
from reduce_session.git_ops import (
    ensure_git_repo, git_snapshot, session_short_id,
    make_tag, get_reduction_tags, find_backups, do_apply, do_restore,
)


def test_ensure_git_repo_creates_repo(tmp_path):
    result = ensure_git_repo(str(tmp_path))
    assert result is True
    assert (tmp_path / ".git").is_dir()
    assert (tmp_path / ".gitignore").exists()


def test_ensure_git_repo_idempotent(tmp_path):
    ensure_git_repo(str(tmp_path))
    result = ensure_git_repo(str(tmp_path))
    assert result is False  # already existed


def test_session_short_id():
    assert session_short_id("/path/to/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl") == "db776eab"
    assert session_short_id("/path/to/db776eab-e7c2-4e9d-8855-28294c27b5db.20260319.jsonl") == "db776eab"


def test_git_snapshot_creates_tag(tmp_path):
    ensure_git_repo(str(tmp_path))
    test_file = tmp_path / "test.jsonl"
    test_file.write_text('{"type":"user"}\n')
    sha = git_snapshot(str(tmp_path), "test.jsonl", "test/tag", "test commit")
    assert sha is not None
    tags = get_reduction_tags(str(tmp_path))
    assert "test/tag" in tags
```

- [ ] **Step 2: Run tests — expect failure**

```bash
uv run pytest tests/test_git_ops.py -v
```

- [ ] **Step 3: Create git_ops.py by extracting from cli.py**

Extract: `GITIGNORE_CONTENT` (line 166), `_run_git` (190), `ensure_git_repo` (205), `_write_gitignore` (228), `_update_gitignore` (234), `git_snapshot` (242), `git_restore_from_tag` (257), `session_short_id` (263), `make_tag` (270), `get_reduction_tags` (277), `get_tag_file_size` (288), `do_init` (883), `find_backups` (925), `do_restore` (934), `do_apply` (976), `do_history` (1051).

Note: `do_apply` imports from `reduction.py` for the staleness check. Add required imports (`os`, `sys`, `glob`, `shutil`, `subprocess`, `datetime`).

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/test_git_ops.py -v
```

- [ ] **Step 5: Update cli.py to import from git_ops.py**

cli.py should now be ~100-150 lines: just `parse_args()`, `main()`, and the output formatting logic.

- [ ] **Step 6: Verify CLI still works**

```bash
uv run reduce-session --dry-run --tokens ~/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl
```

- [ ] **Step 7: Run all tests**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 8: Commit**

```bash
git add src/reduce_session/git_ops.py src/reduce_session/cli.py tests/test_git_ops.py
git commit -m "refactor: extract git operations into git_ops.py"
```

---

### Task 3: Session discovery module

**Files:**
- Create: `src/reduce_session/session.py`
- Create: `tests/test_session.py`

- [ ] **Step 1: Write tests for session discovery**

```python
# tests/test_session.py
import json
from pathlib import Path
from reduce_session.session import (
    SessionInfo, Exchange, scan_projects, parse_tail,
    format_age, derive_project_name,
)


def test_derive_project_name():
    assert derive_project_name("-Users-rwaugh-src-mine-ripvec") == "ripvec"
    assert derive_project_name("-Users-rwaugh-src-mine-ShopifyQuickbooksBridge") == "ShopifyQuickbooksBridge"


def test_format_age():
    from datetime import datetime, timedelta
    now = datetime.now()
    assert format_age(now - timedelta(hours=2)) == "2h"
    assert format_age(now - timedelta(days=3)) == "3d"
    assert format_age(now - timedelta(days=14)) == "14d"


def test_parse_tail_extracts_exchanges(sample_session):
    exchanges, token_est, last_ts = parse_tail(sample_session)
    assert len(exchanges) > 0
    assert any(e.role == "user" for e in exchanges)
    assert any(e.role == "assistant" for e in exchanges)
    assert token_est > 0  # from message.usage
    assert last_ts is not None


def test_parse_tail_handles_corrupt_json(tmp_path):
    bad = tmp_path / "corrupt.jsonl"
    bad.write_text('{"valid": true}\nthis is not json\n{"also": "valid"}\n')
    exchanges, token_est, last_ts = parse_tail(bad)
    # Should not raise, returns what it can


def test_parse_tail_handles_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    exchanges, token_est, last_ts = parse_tail(empty)
    assert exchanges == []
    assert token_est == 0


def test_scan_projects(sample_project_dir, tmp_path):
    projects_dir = tmp_path / "projects"
    sessions = scan_projects(projects_dir)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.project_name == "myproject"
    assert len(s.short_id) == 8
    assert s.size_bytes > 0
    assert s.parse_error is False


def test_scan_projects_skips_bak_files(sample_project_dir, tmp_path):
    # Create a .bak file alongside the session
    import shutil
    for f in sample_project_dir.glob("*.jsonl"):
        shutil.copy(f, f.with_suffix(".jsonl.bak"))
    sessions = scan_projects(tmp_path / "projects")
    assert len(sessions) == 1  # still only one, .bak ignored


def test_continuation_file_grouping(sample_project_dir, tmp_path):
    # Create a continuation file
    for f in sample_project_dir.glob("*.jsonl"):
        uuid_part = f.stem
        cont = f.parent / f"{uuid_part}.20260319_171901.jsonl"
        cont.write_text('{"type":"user","message":{"content":"cont"}}\n')
    sessions = scan_projects(tmp_path / "projects")
    assert len(sessions) == 1  # grouped, not separate
    assert len(sessions[0].continuation_files) == 1
```

- [ ] **Step 2: Run tests — expect failure**

```bash
uv run pytest tests/test_session.py -v
```

- [ ] **Step 3: Create session.py**

```python
# src/reduce_session/session.py
"""Session discovery and metadata extraction for Claude Code JSONL files."""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Exchange:
    role: str              # "user", "assistant", "tool"
    text: str
    tool_name: str | None = None
    tool_status: str | None = None
    is_error: bool = False


@dataclass
class SessionInfo:
    path: Path
    project_name: str
    session_id: str
    short_id: str
    size_bytes: int
    token_estimate: int
    last_timestamp: datetime | None
    age_display: str
    line_count: int
    continuation_files: list[Path] = field(default_factory=list)
    last_exchanges: list[Exchange] = field(default_factory=list)
    parse_error: bool = False


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
CONTINUATION_RE = re.compile(
    r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.\d+.*\.jsonl$"
)


def derive_project_name(slug: str) -> str:
    """Extract readable project name from Claude's directory slug.

    -Users-rwaugh-src-mine-ripvec -> ripvec
    Uses last path component. TODO: handle collisions in v2.
    """
    parts = slug.strip("-").split("-")
    # Reverse-map: the slug is the path with / replaced by -
    # Take the last meaningful component
    return parts[-1] if parts else slug


def format_age(timestamp: datetime) -> str:
    """Format age as relative string: '4h', '2d', '14d'."""
    delta = datetime.now() - timestamp
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)}m"
    if hours < 24:
        return f"{int(hours)}h"
    return f"{int(hours / 24)}d"


def parse_tail(path: Path, tail_bytes: int = 50 * 1024) -> tuple[list[Exchange], int, datetime | None]:
    """Read the tail of a session file and extract exchanges + metadata.

    Returns (exchanges, token_estimate, last_timestamp).
    Handles corrupt JSON, truncated lines, and empty files gracefully.
    """
    exchanges = []
    token_estimate = 0
    last_timestamp = None

    try:
        file_size = path.stat().st_size
    except OSError:
        return exchanges, token_estimate, last_timestamp

    if file_size == 0:
        return exchanges, token_estimate, last_timestamp

    try:
        with open(path, "rb") as f:
            read_size = min(file_size, tail_bytes)
            if file_size > read_size:
                f.seek(file_size - read_size)
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return exchanges, token_estimate, last_timestamp

    lines = raw.strip().split("\n")
    # Skip first line if we seeked (may be truncated)
    if file_size > tail_bytes:
        lines = lines[1:]

    # Walk lines to extract exchanges and metadata
    # Track tool_use IDs to pair with results
    pending_tools = {}  # tool_use_id -> (name, command_preview)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type", "")
        ts = obj.get("timestamp")
        if ts:
            try:
                last_timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
            except (ValueError, AttributeError):
                pass

        # Extract token estimate from assistant usage
        if msg_type == "assistant":
            usage = obj.get("message", {}).get("usage")
            if usage and isinstance(usage, dict):
                inp = usage.get("input_tokens", 0) or 0
                cache_read = usage.get("cache_read_input_tokens", 0) or 0
                cache_create = usage.get("cache_creation_input_tokens", 0) or 0
                total = inp + cache_read + cache_create
                if total > 0:
                    token_estimate = total

        # Skip non-conversation types
        if msg_type in ("progress", "file-history-snapshot", "queue-operation",
                        "last-prompt", "system"):
            continue

        inner = obj.get("message", {})
        content = inner.get("content", "")

        # User text prompt
        if msg_type == "user" and isinstance(content, str) and content.strip():
            # Skip task notifications and local commands
            if "<task-notification>" in content or "<local-command" in content:
                continue
            exchanges.append(Exchange(role="user", text=content.strip()[:500]))

        # User message with content blocks (tool results)
        if msg_type == "user" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type", "")
                if bt == "text":
                    text = block.get("text", "").strip()
                    if text and "<task-notification>" not in text and "<local-command" not in text:
                        exchanges.append(Exchange(role="user", text=text[:500]))
                elif bt == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    is_err = bool(block.get("is_error"))
                    result_text = ""
                    inner_content = block.get("content", "")
                    if isinstance(inner_content, str):
                        result_text = inner_content.strip().split("\n")[0][:80]
                    elif isinstance(inner_content, list):
                        for item in inner_content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                result_text = item.get("text", "").strip().split("\n")[0][:80]
                                break

                    if tool_id in pending_tools:
                        name, preview = pending_tools.pop(tool_id)
                        status = f"error: {result_text}" if is_err else (result_text or "ok")
                        exchanges.append(Exchange(
                            role="tool", text=f"[{name}: {preview}] -> {status}",
                            tool_name=name, tool_status=status, is_error=is_err,
                        ))

        # Assistant messages
        if msg_type == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type", "")
                if bt == "text":
                    text = block.get("text", "").strip()
                    if text:
                        exchanges.append(Exchange(role="assistant", text=text[:500]))
                elif bt == "tool_use":
                    name = block.get("name", "unknown")
                    tool_id = block.get("id", "")
                    inp = block.get("input", {})
                    # Build a short preview
                    if name == "Bash":
                        preview = (inp.get("command", "") if isinstance(inp, dict) else "")[:60]
                    elif name in ("Read", "Edit", "Write"):
                        preview = (inp.get("file_path", "") if isinstance(inp, dict) else "")
                    elif name == "Agent":
                        preview = (inp.get("description", "") if isinstance(inp, dict) else "")[:40]
                    else:
                        # MCP or other tools — shorten the name
                        preview = name.split("__")[-1] if "__" in name else ""
                        name = name.split("__")[-1] if "__" in name else name
                    if tool_id:
                        pending_tools[tool_id] = (name, preview)

    # Heuristic fallback if no usage found
    if token_estimate == 0 and file_size > 0:
        token_estimate = file_size // 14

    return exchanges, token_estimate, last_timestamp


def _count_lines(path: Path) -> int:
    """Count lines without parsing — just count newlines."""
    count = 0
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                count += chunk.count(b"\n")
    except OSError:
        pass
    return count


def scan_projects(projects_dir: Path) -> list[SessionInfo]:
    """Scan all project directories for session files.

    Returns a flat list of SessionInfo sorted by project name, then newest first.
    """
    sessions = []

    if not projects_dir.is_dir():
        return sessions

    try:
        project_dirs = sorted(projects_dir.iterdir())
    except PermissionError:
        return sessions

    for pdir in project_dirs:
        if not pdir.is_dir():
            continue

        project_name = derive_project_name(pdir.name)

        try:
            jsonl_files = list(pdir.glob("*.jsonl"))
        except PermissionError:
            continue

        # Separate main files from continuations
        main_files = {}  # uuid -> path
        continuations = {}  # uuid -> [paths]

        for f in jsonl_files:
            name = f.name
            if name.endswith(".bak") or name.endswith(".reduced"):
                continue

            # Check if it's a continuation (UUID.TIMESTAMP.jsonl)
            m = CONTINUATION_RE.match(name)
            if m:
                uuid = m.group(1)
                continuations.setdefault(uuid, []).append(f)
                continue

            # Main file: UUID.jsonl
            stem = f.stem
            if UUID_RE.match(stem):
                main_files[stem] = f

        for uuid, path in main_files.items():
            try:
                stat = path.stat()
            except OSError:
                continue
            if stat.st_size == 0:
                continue

            parse_error = False
            try:
                exchanges, token_est, last_ts = parse_tail(path)
            except Exception:
                exchanges, token_est, last_ts = [], 0, None
                parse_error = True

            age = format_age(last_ts) if last_ts else "?"
            cont_files = sorted(continuations.get(uuid, []))

            sessions.append(SessionInfo(
                path=path,
                project_name=project_name,
                session_id=uuid,
                short_id=uuid[:8],
                size_bytes=stat.st_size,
                token_estimate=token_est,
                last_timestamp=last_ts,
                age_display=age,
                line_count=_count_lines(path),
                continuation_files=cont_files,
                last_exchanges=exchanges,
                parse_error=parse_error,
            ))

    # Sort: by project name, then newest first within project
    sessions.sort(key=lambda s: (s.project_name.lower(), -(s.last_timestamp.timestamp() if s.last_timestamp else 0)))
    return sessions
```

- [ ] **Step 4: Run tests — expect pass**

```bash
uv run pytest tests/test_session.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/reduce_session/session.py tests/test_session.py
git commit -m "feat: add session discovery module"
```

---

### Task 4: Add textual dependency and styles.tcss

**Files:**
- Modify: `pyproject.toml`
- Create: `src/reduce_session/styles.tcss`

- [ ] **Step 1: Update pyproject.toml**

```toml
[project]
name = "reduce-session"
version = "0.2.0"
description = "Reduce Claude Code session JSONL files while preserving conversation quality"
requires-python = ">=3.10"
dependencies = ["textual>=1.0"]

[project.scripts]
reduce-session = "reduce_session.cli:main"

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create styles.tcss**

Consult context7 for textual CSS syntax. This defines the split layout, colors, and widget styling.

```css
/* src/reduce_session/styles.tcss */

/* Main layout: left panel 40%, right panel 60% */
#main-container {
    layout: horizontal;
    height: 1fr;
}

#session-list {
    width: 40%;
    min-width: 30;
    border-right: tall $primary-background-lighten-2;
}

#preview-panel {
    width: 60%;
    min-width: 40;
}

/* Session tree */
#session-tree {
    height: 1fr;
    scrollbar-gutter: stable;
}

#aggregate-stats {
    height: 1;
    dock: bottom;
    background: $primary-background-lighten-1;
    color: $text-muted;
    padding: 0 1;
}

/* Preview pane */
#empty-state {
    height: 1fr;
    content-align: center middle;
    color: $text-muted;
}

#info-bar {
    height: 3;
    padding: 0 1;
    background: $primary-background-lighten-1;
}

#conversation-log {
    height: 1fr;
    padding: 0 1;
    overflow-y: auto;
}

/* Token gauge colors */
.token-green {
    color: #00d4aa;
}

.token-yellow {
    color: #ffd700;
}

.token-orange {
    color: #ff8c00;
}

.token-red {
    color: #ff4444;
}

/* Conversation roles */
.role-user {
    color: #6ec1e4;
}

.role-assistant {
    color: #e88a6a;
}

.role-tool {
    color: $text-muted;
}

.role-tool-error {
    color: #ff4444;
}

/* Reduce Modal */
ReduceModal {
    align: center middle;
}

#reduce-modal-container {
    width: 80;
    max-width: 90%;
    height: auto;
    max-height: 85%;
    border: thick $accent;
    background: $surface;
    padding: 1 2;
    overflow-y: auto;
}

#modal-title {
    text-style: bold;
    width: 100%;
    content-align: center middle;
    padding-bottom: 1;
}

.modal-section-header {
    color: $accent;
    text-style: bold;
    padding-top: 1;
}

#profile-bar {
    layout: horizontal;
    height: 1;
    padding: 0 0 1 0;
}

.profile-btn {
    width: auto;
    min-width: 12;
    margin: 0 1 0 0;
}

.profile-btn.-active {
    background: $accent;
    color: $text;
    text-style: bold;
}

#modal-actions {
    layout: horizontal;
    height: 3;
    align: right middle;
    padding-top: 1;
}

#modal-actions Button {
    margin: 0 0 0 1;
}

#spinner-container {
    height: 3;
    content-align: center middle;
}
```

- [ ] **Step 3: Verify install picks up textual**

```bash
uv run python -c "import textual; print(textual.__version__)"
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/reduce_session/styles.tcss uv.lock
git commit -m "feat: add textual dependency and TUI stylesheet"
```

---

### Task 5: Token gauge and conversation preview widgets

**Files:**
- Create: `src/reduce_session/widgets.py`

- [ ] **Step 1: Create widgets.py with TokenGauge and ConversationPreview**

Consult context7 for `Static` widget API, Rich `Text` markup, and compose patterns.

```python
# src/reduce_session/widgets.py
"""Custom widgets for the session browser TUI."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from .session import Exchange, SessionInfo


def token_color(tokens: int) -> str:
    """Return a color name based on token pressure."""
    if tokens < 200_000:
        return "#00d4aa"  # green
    elif tokens < 500_000:
        return "#ffd700"  # yellow
    elif tokens < 800_000:
        return "#ff8c00"  # orange
    return "#ff4444"  # red


def render_token_gauge(tokens: int, max_tokens: int = 1_000_000, width: int = 20) -> Text:
    """Render a colored block gauge: ▓▓▓▓▓░░░░░  450k / 1M"""
    ratio = min(tokens / max_tokens, 1.0)
    filled = int(ratio * width)
    empty = width - filled
    color = token_color(tokens)

    gauge = Text()
    gauge.append("▓" * filled, style=color)
    gauge.append("░" * empty, style="dim")
    gauge.append(f"  {tokens // 1000}k / {max_tokens // 1000}k", style="dim")
    return gauge


def render_exchanges(exchanges: list[Exchange]) -> Text:
    """Render conversation exchanges with role coloring."""
    text = Text()
    for ex in exchanges:
        if ex.role == "user":
            text.append("user: ", style="bold #6ec1e4")
            text.append(ex.text + "\n\n", style="#6ec1e4")
        elif ex.role == "assistant":
            text.append("assistant: ", style="bold #e88a6a")
            text.append(ex.text + "\n\n", style="#e88a6a")
        elif ex.role == "tool":
            style = "#ff4444" if ex.is_error else "dim"
            text.append(ex.text + "\n", style=style)
    return text


class InfoBar(Static):
    """Session metadata and token gauge."""

    def update_session(self, session: SessionInfo | None) -> None:
        if session is None:
            self.update("")
            return

        text = Text()
        text.append(f"{session.short_id}", style="bold")
        text.append(f"  ~{session.token_estimate // 1000}k tokens", style="dim")
        text.append(f"  {session.age_display} ago", style="dim")
        text.append(f"  {session.size_bytes / 1024 / 1024:.1f} MB", style="dim")
        text.append(f"  {session.line_count:,} lines\n", style="dim")
        text.append_text(render_token_gauge(session.token_estimate))
        self.update(text)


class ConversationPreview(Static):
    """Scrollable conversation preview with role-colored exchanges."""

    def update_session(self, session: SessionInfo | None) -> None:
        if session is None:
            self.update(Text("Select a session to preview its conversation",
                             style="dim italic", justify="center"))
            return

        if session.parse_error:
            self.update(Text("⚠ Error parsing session file", style="bold red"))
            return

        if not session.last_exchanges:
            self.update(Text("(empty session)", style="dim italic"))
            return

        self.update(render_exchanges(session.last_exchanges))
```

- [ ] **Step 2: Quick test import**

```bash
uv run python -c "from reduce_session.widgets import InfoBar, ConversationPreview, render_token_gauge; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/reduce_session/widgets.py
git commit -m "feat: add token gauge and conversation preview widgets"
```

---

### Task 6: Main TUI app with session tree and preview

**Files:**
- Create: `src/reduce_session/tui.py`
- Modify: `src/reduce_session/cli.py` (add --browse / no-args dispatch)

- [ ] **Step 1: Create tui.py**

Consult context7 for `App`, `Tree`, `Header`, `Footer`, `Horizontal`, `Vertical`, `Binding`, `on` decorator, and `Tree.NodeHighlighted` event.

```python
# src/reduce_session/tui.py
"""Session browser TUI application."""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode

from .session import SessionInfo, scan_projects
from .widgets import ConversationPreview, InfoBar, token_color


def get_projects_dir() -> Path:
    """Return the Claude projects directory."""
    import os
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / "projects"
    return Path.home() / ".claude" / "projects"


class SessionBrowserApp(App):
    """Browse Claude Code sessions and reduce them."""

    CSS_PATH = "styles.tcss"
    TITLE = "reduce-session"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "reduce", "Reduce", show=True),
        Binding("d", "dry_run", "Dry Run", show=True),
        Binding("shift+r", "refresh", "Refresh", show=True, key_display="R"),
    ]

    def __init__(self, projects_dir: Path | None = None):
        super().__init__()
        self.projects_dir = projects_dir or get_projects_dir()
        self.sessions: list[SessionInfo] = []
        self.selected_session: SessionInfo | None = None
        self._node_to_session: dict[int, SessionInfo] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            with Vertical(id="session-list"):
                yield Tree("Projects", id="session-tree")
                yield Static("", id="aggregate-stats")
            with Vertical(id="preview-panel"):
                yield InfoBar("", id="info-bar")
                yield ConversationPreview(
                    Text("Select a session to preview its conversation",
                         style="dim italic", justify="center"),
                    id="conversation-log",
                )
        yield Footer()

    def on_mount(self) -> None:
        self._load_sessions()

    def _load_sessions(self) -> None:
        self.sessions = scan_projects(self.projects_dir)
        tree = self.query_one("#session-tree", Tree)
        tree.clear()
        self._node_to_session.clear()

        # Group by project
        projects: dict[str, list[SessionInfo]] = {}
        for s in self.sessions:
            projects.setdefault(s.project_name, []).append(s)

        for proj_name in sorted(projects.keys()):
            proj_sessions = projects[proj_name]
            proj_node = tree.root.add(proj_name, expand=True)
            for s in proj_sessions:
                color = token_color(s.token_estimate)
                label = Text()
                label.append(f"{s.short_id}", style="bold")
                label.append(f"  ~{s.token_estimate // 1000}k tok", style="dim")
                label.append(f"  {s.age_display}", style="dim")
                if s.parse_error:
                    label.append("  ⚠", style="bold yellow")
                else:
                    label.append("  ●", style=color)
                node = proj_node.add_leaf(label)
                self._node_to_session[id(node)] = s

        # Aggregate stats
        total_sessions = len(self.sessions)
        total_projects = len(projects)
        total_size = sum(s.size_bytes for s in self.sessions)
        stats = self.query_one("#aggregate-stats", Static)
        stats.update(f"{total_projects} projects  {total_sessions} sessions  {total_size / 1024 / 1024:.1f} MB")

        tree.root.expand()

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        session = self._node_to_session.get(id(event.node))
        self.selected_session = session
        self.query_one("#info-bar", InfoBar).update_session(session)
        self.query_one("#conversation-log", ConversationPreview).update_session(session)

    def action_reduce(self) -> None:
        if self.selected_session:
            self.push_screen(
                "reduce_modal",
                ReduceModal(self.selected_session, read_only=False),
            )

    def action_dry_run(self) -> None:
        if self.selected_session:
            self.push_screen(
                "reduce_modal",
                ReduceModal(self.selected_session, read_only=True),
            )

    def action_refresh(self) -> None:
        self._load_sessions()


# Import here to avoid circular — ReduceModal is defined in widgets.py
# but needs to be registered. We'll use lazy import in action_reduce/dry_run.
# For now, placeholder:
class ReduceModal:
    """Placeholder — implemented in Task 7."""
    pass
```

- [ ] **Step 2: Update cli.py to launch TUI when no args**

Add `--browse` flag. When no positional arg is given, launch the TUI.

```python
# In parse_args(), make input optional:
p.add_argument("input", nargs="?", default=None, help="Path to session JSONL file")
p.add_argument("--browse", action="store_true", help="Launch interactive session browser")

# In main(), at the top:
if args.browse or (args.input is None and not any([args.restore, args.history, args.init])):
    from .tui import SessionBrowserApp
    app = SessionBrowserApp()
    app.run()
    return
```

- [ ] **Step 3: Test TUI launches**

```bash
uv run reduce-session --browse
```

Should show the session tree with real data from `~/.claude/projects/`. Press `q` to quit.

- [ ] **Step 4: Test no-args also launches TUI**

```bash
uv run reduce-session
```

- [ ] **Step 5: Test existing CLI still works**

```bash
uv run reduce-session --dry-run --tokens ~/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl
```

- [ ] **Step 6: Commit**

```bash
git add src/reduce_session/tui.py src/reduce_session/cli.py
git commit -m "feat: add TUI session browser with tree and preview"
```

---

### Task 7: Reduce modal with worker threading

**Files:**
- Modify: `src/reduce_session/widgets.py`
- Modify: `src/reduce_session/tui.py` (replace placeholder)

- [ ] **Step 1: Add ReduceModal to widgets.py**

Consult context7 for `ModalScreen`, `Worker`, `run_worker`, `Worker.StateChanged`, `Button`, `LoadingIndicator`.

```python
# Add to widgets.py:

import os
from textual.screen import ModalScreen
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, LoadingIndicator
from textual.worker import Worker, WorkerState

from .reduction import reduce_session, ReductionResult
from .git_ops import do_apply, ensure_git_repo


class ReduceModal(ModalScreen):
    """Modal overlay showing reduction results with Apply/Cancel."""

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("g", "profile_gentle", "Gentle"),
        ("s", "profile_standard", "Standard"),
        ("a", "profile_aggressive", "Aggressive"),
        ("enter", "apply", "Apply"),
    ]

    def __init__(self, session: SessionInfo, read_only: bool = False):
        super().__init__()
        self.session = session
        self.read_only = read_only
        self.current_profile = "standard"
        self.result: ReductionResult | None = None
        self.source_mtime: float | None = None
        self._current_worker: Worker | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="reduce-modal-container"):
            title = f"Reduce: {self.session.short_id} ({self.session.project_name})"
            if self.read_only:
                title += " [dry-run]"
            yield Static(title, id="modal-title")

            with Horizontal(id="profile-bar"):
                yield Button("gentle", id="btn-gentle", classes="profile-btn")
                yield Button("standard", id="btn-standard", classes="profile-btn -active")
                yield Button("aggressive", id="btn-aggressive", classes="profile-btn")
                yield Static("  Cut: 50%  Fade: 75%", id="cut-fade-display")

            yield Static("Running reduction...", id="dry-run-stats")
            yield Static("", id="token-viz")
            yield Static("", id="strategies-grid")
            yield Static("", id="safety-checks")
            yield LoadingIndicator(id="spinner-container")

            with Horizontal(id="modal-actions"):
                if not self.read_only:
                    yield Button("Apply", variant="primary", id="btn-apply")
                yield Button("Cancel", variant="default", id="btn-cancel")

    def on_mount(self) -> None:
        self._run_reduction(self.current_profile)

    def _run_reduction(self, profile: str) -> None:
        # Cancel any in-flight worker
        if self._current_worker and self._current_worker.state == WorkerState.RUNNING:
            self._current_worker.cancel()

        self.current_profile = profile
        self._update_profile_buttons()
        self.query_one("#dry-run-stats", Static).update("Running reduction...")
        self.query_one("#token-viz", Static).update("")
        self.query_one("#strategies-grid", Static).update("")
        self.query_one("#safety-checks", Static).update("")

        try:
            self.query_one("#spinner-container").display = True
        except Exception:
            pass

        self.source_mtime = os.path.getmtime(str(self.session.path))
        self._current_worker = self.run_worker(
            self._do_reduce(profile), exclusive=True
        )

    async def _do_reduce(self, profile: str) -> ReductionResult:
        return reduce_session(
            str(self.session.path),
            profile=profile,
            estimate_tokens=True,
        )

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state == WorkerState.SUCCESS and event.worker.result:
            self.result = event.worker.result
            self._render_results()
        elif event.state == WorkerState.ERROR:
            self.query_one("#dry-run-stats", Static).update(
                Text(f"Error: {event.worker.error}", style="bold red")
            )
        try:
            self.query_one("#spinner-container").display = (
                event.state == WorkerState.RUNNING
            )
        except Exception:
            pass

    def _render_results(self) -> None:
        r = self.result
        if not r:
            return

        # Dry run stats
        saved_pct = (r.orig_size - r.new_size) / r.orig_size * 100 if r.orig_size else 0
        stats = Text()
        stats.append("── Dry Run Results ──\n\n", style="bold #00d4aa")
        stats.append(f"  Original:  {r.orig_count:>6,} lines   {r.orig_size / 1024 / 1024:.1f} MB\n")
        stats.append(f"  Reduced:   {r.new_count:>6,} lines   {r.new_size / 1024 / 1024:.1f} MB\n")
        stats.append(f"  Saved:     {r.orig_count - r.new_count:>6,} lines   ")
        stats.append(f"{(r.orig_size - r.new_size) / 1024 / 1024:.1f} MB  ({saved_pct:.0f}%)\n",
                     style="bold #00d4aa")
        self.query_one("#dry-run-stats", Static).update(stats)

        # Token estimate
        token_text = Text()
        token_text.append("── Token Estimate ──\n\n", style="bold #00d4aa")
        if r.orig_budget and r.reduced_budget:
            orig_tok = r.orig_budget.context_total
            red_tok = r.reduced_budget.context_total
            if r.api_tokens and r.orig_budget._raw_chars > 0:
                cpt = r.orig_budget._raw_chars / r.api_tokens
                orig_tok = r.api_tokens
                red_tok = int(r.reduced_budget._raw_chars / cpt)
                token_text.append(f"  (calibrated from API: {cpt:.1f} chars/tok)\n", style="dim")
            token_text.append("  Before: ")
            token_text.append_text(render_token_gauge(orig_tok))
            token_text.append("\n  After:  ")
            token_text.append_text(render_token_gauge(red_tok))
            token_text.append("\n")
        self.query_one("#token-viz", Static).update(token_text)

        # Strategies
        strat_text = Text()
        strat_text.append("── Strategies Applied ──\n\n", style="bold #00d4aa")
        items = sorted(r.stats.items(), key=lambda x: -x[1])
        col1 = items[:len(items)//2 + 1]
        col2 = items[len(items)//2 + 1:]
        for i in range(max(len(col1), len(col2))):
            if i < len(col1):
                name, count = col1[i]
                strat_text.append(f"  {name:<28s} {count:>5,}")
            if i < len(col2):
                name, count = col2[i]
                strat_text.append(f"    {name:<28s} {count:>5,}")
            strat_text.append("\n")
        self.query_one("#strategies-grid", Static).update(strat_text)

        # Safety
        safety_text = Text()
        safety_text.append("── Safety ──\n\n", style="bold #00d4aa")
        has_git = os.path.isdir(os.path.join(os.path.dirname(str(self.session.path)), ".git"))
        safety_text.append(f"  ✓ parentUuid chain preserved ({r.stats.get('reparented', 0)} reparented)\n", style="#00d4aa")
        if has_git:
            safety_text.append("  ✓ git repo initialized\n", style="#00d4aa")
        else:
            safety_text.append("  ○ no git repo (run reduce-session --init)\n", style="dim")
        safety_text.append("  ✓ .bak safety net will be created\n", style="#00d4aa")
        if self.session.continuation_files:
            safety_text.append(f"  ℹ {len(self.session.continuation_files)} continuation file(s) — main file only\n", style="#ffd700")
        self.query_one("#safety-checks", Static).update(safety_text)

    def _update_profile_buttons(self) -> None:
        for profile in ("gentle", "standard", "aggressive"):
            btn = self.query_one(f"#btn-{profile}", Button)
            btn.set_classes("profile-btn" + (" -active" if profile == self.current_profile else ""))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-apply":
            self._do_apply()
        elif event.button.id == "btn-cancel":
            self.dismiss()
        elif event.button.id in ("btn-gentle", "btn-standard", "btn-aggressive"):
            profile = event.button.id.replace("btn-", "")
            self._run_reduction(profile)

    def action_cancel(self) -> None:
        self.dismiss()

    def action_apply(self) -> None:
        if not self.read_only:
            self._do_apply()

    def action_profile_gentle(self) -> None:
        self._run_reduction("gentle")

    def action_profile_standard(self) -> None:
        self._run_reduction("standard")

    def action_profile_aggressive(self) -> None:
        self._run_reduction("aggressive")

    def _do_apply(self) -> None:
        if not self.result:
            return

        # Staleness check
        current_mtime = os.path.getmtime(str(self.session.path))
        if self.source_mtime and current_mtime > self.source_mtime:
            self.query_one("#safety-checks", Static).update(
                Text("⚠ Session file was modified since analysis.\n  Re-run reduction first.",
                     style="bold red")
            )
            return

        # Write .reduced file
        output_path = str(self.session.path) + ".reduced"
        with open(output_path, "w") as f:
            f.writelines(self.result.kept_lines)

        # Apply with git + .bak
        try:
            do_apply(str(self.session.path), output_path,
                     self.current_profile, 50, 75)
            self.dismiss(True)  # True signals success to parent
        except Exception as e:
            self.query_one("#safety-checks", Static).update(
                Text(f"⚠ Apply failed: {e}", style="bold red")
            )
```

- [ ] **Step 2: Update tui.py to import real ReduceModal**

Replace the placeholder class and update the `action_reduce`/`action_dry_run` methods:

```python
# In tui.py, replace the placeholder:
from .widgets import ConversationPreview, InfoBar, ReduceModal, token_color

# Update action_reduce:
def action_reduce(self) -> None:
    if self.selected_session:
        self.push_screen(ReduceModal(self.selected_session, read_only=False),
                         callback=self._on_modal_dismiss)

def action_dry_run(self) -> None:
    if self.selected_session:
        self.push_screen(ReduceModal(self.selected_session, read_only=True))

def _on_modal_dismiss(self, applied: bool | None) -> None:
    if applied:
        self._load_sessions()  # refresh after successful apply
```

- [ ] **Step 3: Test the full TUI with modal**

```bash
uv run reduce-session
```

Navigate to a session, press `r`, verify the modal opens with reduction results. Press `Esc` to cancel. Try `d` for dry-run mode (no Apply button). Try switching profiles with `g`/`s`/`a`.

- [ ] **Step 4: Test Apply on a real session (use a copy)**

```bash
cp ~/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl /tmp/test-apply.jsonl
```

Then test in the TUI — the Apply should create git tags and .bak.

- [ ] **Step 5: Commit**

```bash
git add src/reduce_session/widgets.py src/reduce_session/tui.py
git commit -m "feat: add reduce modal with worker threading and apply"
```

---

### Task 8: Visual polish and edge cases

**Files:**
- Modify: `src/reduce_session/styles.tcss`
- Modify: `src/reduce_session/widgets.py`
- Modify: `src/reduce_session/tui.py`

- [ ] **Step 1: Refine tree node styling**

Ensure dimming for old sessions (>7d), parse error indicators, and token dot colors render correctly. Verify with real session data.

- [ ] **Step 2: Handle empty projects directory**

If `~/.claude/projects/` doesn't exist or is empty, show a helpful message in the tree area.

- [ ] **Step 3: Handle narrow terminal**

Test at 80 columns. If the split panel is too cramped, consider a minimum width warning or auto-adjusting the split ratio.

- [ ] **Step 4: Keyboard navigation polish**

Verify `j`/`k` work alongside arrow keys. Verify `Enter` on project nodes expands/collapses. Verify `Enter` on session nodes opens the modal.

- [ ] **Step 5: Test with multiple projects**

Verify the tree groups sessions correctly, sorts by project name then newest-first.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "polish: visual refinements and edge case handling"
```

---

### Task 9: Final integration testing and push

**Files:**
- All files

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 2: Test the full workflow end-to-end**

1. `uv run reduce-session` — TUI opens
2. Navigate to a session, verify preview
3. Press `d` — dry-run modal, verify stats, press `Esc`
4. Press `r` — reduce modal, switch profiles with `g`/`s`/`a`
5. Press `Esc` — cancel
6. Test existing CLI still works: `uv run reduce-session --dry-run --tokens <file>`
7. Test `--help` works

- [ ] **Step 3: Commit any remaining changes**

```bash
git add -A
git commit -m "feat: session browser TUI complete"
```

- [ ] **Step 4: Push**

```bash
git push
```
