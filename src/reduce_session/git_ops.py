"""Git-based preservation for session reduction history.

Provides git init, snapshot, tag, restore, and history operations
that both the CLI and TUI can use. All functions raise exceptions
on error instead of calling sys.exit(), and return structured data
instead of printing.
"""

import glob
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime


GITIGNORE_CONTENT = """\
# Session reduction backups (superseded by git history)
*.bak
*.bak2
*.reduced

# Per-session runtime directories
*/subagents/
*/tool-results/

# Only track *.jsonl session files
*.md
*.py
*.tar.gz
.claude/
.ruff_cache/
memory/
reduce-session/
"""


# --- Git primitives ---


def _run_git(project_dir, *args, check=True):
    """Run a git command in the project directory, suppressing output."""
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=check,
        )
        return r
    except FileNotFoundError:
        return None  # git not installed


def ensure_git_repo(project_dir):
    """Initialize git repo in project dir if needed. Returns True if newly created."""
    git_dir = os.path.join(project_dir, ".git")
    if os.path.isdir(git_dir):
        _update_gitignore(project_dir)
        return False

    r = _run_git(project_dir, "init")
    if r is None:
        return False

    _write_gitignore(project_dir)

    # Initial commit of all existing .jsonl files
    jsonl_files = glob.glob(os.path.join(project_dir, "*.jsonl"))
    if jsonl_files:
        _run_git(project_dir, "add", *[os.path.basename(f) for f in jsonl_files])
    _run_git(project_dir, "add", ".gitignore")
    _run_git(project_dir, "commit", "-m", "init: track session files", check=False)
    return True


def _write_gitignore(project_dir):
    path = os.path.join(project_dir, ".gitignore")
    with open(path, "w") as f:
        f.write(GITIGNORE_CONTENT)


def _update_gitignore(project_dir):
    path = os.path.join(project_dir, ".gitignore")
    if not os.path.exists(path):
        _write_gitignore(project_dir)
        _run_git(project_dir, "add", ".gitignore")
        _run_git(project_dir, "commit", "-m", "add .gitignore", check=False)


def git_snapshot(project_dir, session_basename, tag, message):
    """Stage session file, commit, and optionally tag. Returns commit SHA or None."""
    _run_git(project_dir, "add", session_basename)
    r = _run_git(project_dir, "commit", "-m", message, check=False)
    if r and r.returncode == 0:
        sha = _run_git(project_dir, "rev-parse", "--short", "HEAD")
        if tag:
            _run_git(project_dir, "tag", tag)
        return sha.stdout.strip() if sha else None
    # Nothing to commit (file unchanged)
    if tag:
        _run_git(project_dir, "tag", tag, check=False)
    return None


def git_restore_from_tag(project_dir, tag, session_basename):
    """Restore a single session file from a tag without moving HEAD."""
    r = _run_git(project_dir, "checkout", tag, "--", session_basename)
    return r is not None and r.returncode == 0


# --- Helpers ---


def session_short_id(session_path):
    """Extract first 8 chars of session UUID from filename."""
    basename = os.path.basename(session_path)
    # Handle both UUID.jsonl and UUID.TIMESTAMP.jsonl
    return basename.split(".")[0][:8]


def make_tag(session_path, phase):
    """Create a tag name: reduce/{short}/{timestamp}/{phase}"""
    short = session_short_id(session_path)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"reduce/{short}/{ts}/{phase}", ts


def get_reduction_tags(project_dir, session_id=None):
    """List all reduction tags, optionally filtered by session short ID."""
    r = _run_git(project_dir, "tag", "-l", "reduce/*", check=False)
    if not r or r.returncode != 0:
        return []
    tags = [t.strip() for t in r.stdout.strip().split("\n") if t.strip()]
    if session_id:
        tags = [t for t in tags if t.startswith(f"reduce/{session_id}/")]
    return sorted(tags)


def get_tag_file_size(project_dir, tag, session_basename):
    """Get file size at a given tag."""
    r = _run_git(project_dir, "show", f"{tag}:{session_basename}", check=False)
    if r and r.returncode == 0:
        return len(r.stdout.encode("utf-8"))
    return None


def get_file_tail_at_tag(project_dir, tag, session_basename, tail_bytes=50 * 1024):
    """Get the last tail_bytes of a session file at a given git tag.

    Returns the raw string content, or None if not available.
    """
    r = _run_git(project_dir, "show", f"{tag}:{session_basename}", check=False)
    if r and r.returncode == 0:
        content = r.stdout
        if len(content) > tail_bytes:
            return content[-tail_bytes:]
        return content
    return None


def find_backups(path):
    """Find all .bak files for a session, newest first."""
    patterns = [f"{path}.bak", f"{path}.*.bak", f"{path}*.bak"]
    found = set()
    for pat in patterns:
        found.update(glob.glob(pat))
    return sorted(found, key=lambda p: os.path.getmtime(p), reverse=True)


# --- High-level operations (return structured data, raise on error) ---


@dataclass
class InitResult:
    """Result of do_init."""

    project_dir: str
    newly_created: bool
    jsonl_count: int
    tag_count: int = 0
    initial_commit: str = ""
    uncommitted_lines: list = field(default_factory=list)


@dataclass
class RestoreResult:
    """Result of do_restore."""

    source: str  # "git" or "bak"
    detail: str  # tag name or .bak path
    size_before: int
    size_after: int
    older_snapshots: int = 0


@dataclass
class ApplyResult:
    """Result of do_apply."""

    input_path: str
    orig_size: int
    new_size: int
    pre_tag: str
    post_tag: str
    bak_path: str
    newly_init: bool = False


@dataclass
class ReductionEntry:
    """One reduction in the history timeline."""

    timestamp: str  # raw timestamp string e.g. 20260323_120000
    ts_display: str  # formatted e.g. 2026-03-23 12:00
    description: str
    pre_tag: str | None
    post_tag: str | None
    pre_size: int | None
    post_size: int | None
    saved_bytes: int | None = None
    saved_pct: float | None = None


@dataclass
class HistoryResult:
    """Result of do_history."""

    session_id: str
    has_git: bool
    reductions: list  # list of ReductionEntry
    backups: list  # list of (path, size, mtime)
    current_size: int


def do_init(path):
    """Initialize git repo for session tracking. Returns InitResult or raises RuntimeError."""
    if os.path.isfile(path):
        project_dir = os.path.dirname(os.path.abspath(path))
    elif os.path.isdir(path):
        project_dir = os.path.abspath(path)
    else:
        raise RuntimeError(f"{path} not found")

    if os.path.isdir(os.path.join(project_dir, ".git")):
        _update_gitignore(project_dir)
        r = _run_git(project_dir, "status", "--short", check=False)
        jsonl_count = len(glob.glob(os.path.join(project_dir, "*.jsonl")))
        tag_r = _run_git(project_dir, "tag", "-l", "reduce/*", check=False)
        tag_count = len(
            [t for t in (tag_r.stdout if tag_r else "").split("\n") if t.strip()]
        )
        uncommitted = []
        if r and r.stdout.strip():
            uncommitted = r.stdout.strip().split("\n")[:5]
        return InitResult(
            project_dir=project_dir,
            newly_created=False,
            jsonl_count=jsonl_count,
            tag_count=tag_count,
            uncommitted_lines=uncommitted,
        )

    newly_init = ensure_git_repo(project_dir)
    if not newly_init:
        raise RuntimeError("git not available")

    jsonl_count = len(glob.glob(os.path.join(project_dir, "*.jsonl")))
    initial_commit = ""
    log_r = _run_git(project_dir, "log", "--oneline", "-1", check=False)
    if log_r and log_r.stdout.strip():
        initial_commit = log_r.stdout.strip()

    return InitResult(
        project_dir=project_dir,
        newly_created=True,
        jsonl_count=jsonl_count,
        initial_commit=initial_commit,
    )


def do_restore(path):
    """Restore from git tag (preferred) or most recent .bak file. Returns RestoreResult or raises RuntimeError."""
    project_dir = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path)
    short = session_short_id(path)

    # Try git first
    if os.path.isdir(os.path.join(project_dir, ".git")):
        tags = get_reduction_tags(project_dir, short)
        pre_tags = [t for t in tags if t.endswith("/pre")]
        if pre_tags:
            tag = pre_tags[-1]  # most recent pre-reduction
            size_before = os.path.getsize(path) if os.path.exists(path) else 0
            if git_restore_from_tag(project_dir, tag, basename):
                size_after = os.path.getsize(path)
                # Commit the restore so history stays clean
                git_snapshot(project_dir, basename, None, f"restore {short} from {tag}")
                return RestoreResult(
                    source="git",
                    detail=tag,
                    size_before=size_before,
                    size_after=size_after,
                    older_snapshots=len(pre_tags) - 1,
                )

    # Fall back to .bak files
    backups = find_backups(path)
    if not backups:
        raise RuntimeError(f"No backups found for {path}")
    newest = backups[0]
    size_before = os.path.getsize(path) if os.path.exists(path) else 0
    size_backup = os.path.getsize(newest)
    shutil.copy2(newest, path)
    return RestoreResult(
        source="bak",
        detail=newest,
        size_before=size_before,
        size_after=size_backup,
    )


def do_apply(input_path, reduced_path, profile_name="standard", cut=50, fade=75):
    """Replace original with reduced, using git for history and .bak as safety net. Returns ApplyResult or raises RuntimeError."""
    if not os.path.exists(reduced_path):
        raise RuntimeError(
            f"No reduced file found at {reduced_path}. "
            "Run without --apply first to generate the reduced file."
        )

    # Safety: refuse if source file changed since the .reduced was generated.
    source_mtime = os.path.getmtime(input_path)
    reduced_mtime = os.path.getmtime(reduced_path)

    if source_mtime > reduced_mtime:
        source_size = os.path.getsize(input_path)
        reduced_size = os.path.getsize(reduced_path)
        raise RuntimeError(
            f"Source file was modified after the .reduced file was generated.\n"
            f"  source:  {datetime.fromtimestamp(source_mtime):%Y-%m-%d %H:%M:%S} ({source_size / 1024 / 1024:.1f} MB)\n"
            f"  reduced: {datetime.fromtimestamp(reduced_mtime):%Y-%m-%d %H:%M:%S} ({reduced_size / 1024 / 1024:.1f} MB)\n"
            "The session may have new messages since reduction. Re-run without --apply first."
        )

    project_dir = os.path.dirname(os.path.abspath(input_path))
    basename = os.path.basename(input_path)
    short = session_short_id(input_path)

    # Git: snapshot pre-reduction state
    newly_init = ensure_git_repo(project_dir)

    pre_tag, ts = make_tag(input_path, "pre")
    post_tag = f"reduce/{short}/{ts}/post"

    git_snapshot(project_dir, basename, pre_tag, f"pre-reduction {short}")

    # .bak safety net (always, even with git)
    bak_path = f"{input_path}.{ts}.bak"
    shutil.copy2(input_path, bak_path)

    # Apply reduction
    orig_size = os.path.getsize(input_path)
    shutil.move(reduced_path, input_path)
    new_size = os.path.getsize(input_path)

    # Git: snapshot post-reduction state
    git_snapshot(
        project_dir,
        basename,
        post_tag,
        f"reduce {short}: {profile_name} cut={cut} fade={fade}",
    )

    return ApplyResult(
        input_path=input_path,
        orig_size=orig_size,
        new_size=new_size,
        pre_tag=pre_tag,
        post_tag=post_tag,
        bak_path=bak_path,
        newly_init=newly_init,
    )


def do_history(path):
    """Return reduction timeline for a session. Returns HistoryResult."""
    project_dir = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path)
    short = session_short_id(path)
    current_size = os.path.getsize(path) if os.path.exists(path) else 0

    if not os.path.isdir(os.path.join(project_dir, ".git")):
        # Fall back to listing .bak files
        backups = find_backups(path)
        backup_info = []
        for bak in backups:
            size = os.path.getsize(bak)
            mtime = datetime.fromtimestamp(os.path.getmtime(bak))
            backup_info.append((bak, size, mtime))
        return HistoryResult(
            session_id=short,
            has_git=False,
            reductions=[],
            backups=backup_info,
            current_size=current_size,
        )

    tags = get_reduction_tags(project_dir, short)

    # Group by timestamp
    raw_reductions = {}
    for tag in tags:
        parts = tag.split("/")  # reduce/{short}/{ts}/{phase}
        if len(parts) == 4:
            ts, phase = parts[2], parts[3]
            raw_reductions.setdefault(ts, {})[phase] = tag

    entries = []
    for ts in sorted(raw_reductions.keys()):
        phases = raw_reductions[ts]
        pre_tag = phases.get("pre")
        post_tag = phases.get("post")

        # Get commit message from post tag for profile info
        desc = ""
        if post_tag:
            r = _run_git(project_dir, "log", "-1", "--format=%s", post_tag, check=False)
            if r and r.stdout.strip():
                desc = r.stdout.strip()

        pre_size = (
            get_tag_file_size(project_dir, pre_tag, basename) if pre_tag else None
        )
        post_size = (
            get_tag_file_size(project_dir, post_tag, basename) if post_tag else None
        )

        ts_display = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} {ts[9:11]}:{ts[11:13]}"

        saved_bytes = None
        saved_pct = None
        if pre_size and post_size:
            saved_bytes = pre_size - post_size
            saved_pct = saved_bytes / pre_size * 100

        entries.append(
            ReductionEntry(
                timestamp=ts,
                ts_display=ts_display,
                description=desc,
                pre_tag=pre_tag,
                post_tag=post_tag,
                pre_size=pre_size,
                post_size=post_size,
                saved_bytes=saved_bytes,
                saved_pct=saved_pct,
            )
        )

    return HistoryResult(
        session_id=short,
        has_git=True,
        reductions=entries,
        backups=[],
        current_size=current_size,
    )
