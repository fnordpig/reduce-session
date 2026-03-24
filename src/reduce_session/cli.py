#!/usr/bin/env python3
"""Reduce a Claude Code session JSONL while preserving conversation quality.

Usage: python3 reduce_session.py [options] <path-to-session.jsonl>

Options:
  --tokens           Print context-window token estimate by category
  --cut PCT          End of fully-aggressive zone (default: 50)
  --fade PCT         Start of fully-gentle zone (default: 75)
  --profile NAME     Preset: gentle, standard (default), aggressive
  --dry-run          Analyze only, don't write output
  --apply            Replace original with reduced (backs up to .bak first)
  --restore          Restore from most recent .bak file
  --chars-per-token  Override chars/token ratio (default: 3.7)

The gradient applies aggressive reduction to messages in [0, cut%],
tapers linearly from aggressive to gentle in [cut%, fade%],
and applies gentle/no reduction in [fade%, 100%].
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime

from reduce_session.reduction import (
    CHARS_PER_TOKEN,
    PROFILES,
    TokenBudget,
    reduce_session,
)

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


# --- Git preservation ---


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
        print("Warning: git not found, falling back to .bak files", file=sys.stderr)
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


# --- CLI ---


def parse_args():
    p = argparse.ArgumentParser(description="Reduce Claude Code session JSONL")
    p.add_argument("input", help="Path to session JSONL file")
    p.add_argument(
        "--tokens", action="store_true", help="Print token estimate by category"
    )
    p.add_argument(
        "--cut",
        type=int,
        default=50,
        help="End of fully-aggressive zone, as %% of conversation (default: 50)",
    )
    p.add_argument(
        "--fade",
        type=int,
        default=75,
        help="Start of fully-gentle zone, as %% of conversation (default: 75)",
    )
    p.add_argument(
        "--profile",
        choices=["gentle", "standard", "aggressive"],
        default="standard",
        help="Limit preset (default: standard)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Analyze only, don't write output"
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Replace original with reduced (creates timestamped .bak first)",
    )
    p.add_argument(
        "--restore",
        action="store_true",
        help="Restore from git tag or most recent .bak",
    )
    p.add_argument(
        "--history",
        action="store_true",
        help="Show reduction timeline for this session",
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="Initialize git repo in the session directory (input is a directory or session file)",
    )
    p.add_argument(
        "--chars-per-token",
        type=float,
        default=CHARS_PER_TOKEN,
        help=f"Chars/token ratio for estimates (default: {CHARS_PER_TOKEN})",
    )
    return p.parse_args()


def do_init(path):
    """Initialize git repo for session tracking in a directory."""
    if os.path.isfile(path):
        project_dir = os.path.dirname(os.path.abspath(path))
    elif os.path.isdir(path):
        project_dir = os.path.abspath(path)
    else:
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    if os.path.isdir(os.path.join(project_dir, ".git")):
        _update_gitignore(project_dir)
        # Show what's tracked
        r = _run_git(project_dir, "status", "--short", check=False)
        jsonl_count = len(glob.glob(os.path.join(project_dir, "*.jsonl")))
        tag_r = _run_git(project_dir, "tag", "-l", "reduce/*", check=False)
        tag_count = len(
            [t for t in (tag_r.stdout if tag_r else "").split("\n") if t.strip()]
        )
        print(f"Git repo already exists: {project_dir}")
        print(f"  {jsonl_count} session file(s)")
        print(f"  {tag_count} reduction tag(s)")
        if r and r.stdout.strip():
            print(f"  uncommitted changes:")
            for line in r.stdout.strip().split("\n")[:5]:
                print(f"    {line}")
        return

    newly_init = ensure_git_repo(project_dir)
    if not newly_init:
        print("Error: git not available", file=sys.stderr)
        sys.exit(1)

    jsonl_count = len(glob.glob(os.path.join(project_dir, "*.jsonl")))
    print(f"Initialized git repo: {project_dir}")
    print(f"  {jsonl_count} session file(s) tracked")
    print(f"  .gitignore configured (tracks *.jsonl only)")
    log_r = _run_git(project_dir, "log", "--oneline", "-1", check=False)
    if log_r and log_r.stdout.strip():
        print(f"  initial commit: {log_r.stdout.strip()}")


def find_backups(path):
    """Find all .bak files for a session, newest first."""
    patterns = [f"{path}.bak", f"{path}.*.bak", f"{path}*.bak"]
    found = set()
    for pat in patterns:
        found.update(glob.glob(pat))
    return sorted(found, key=lambda p: os.path.getmtime(p), reverse=True)


def do_restore(path):
    """Restore from git tag (preferred) or most recent .bak file."""
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
                print(f"Restored from git tag: {tag}")
                print(
                    f"  {size_before / 1024 / 1024:.2f} MB -> {size_after / 1024 / 1024:.2f} MB"
                )
                if len(pre_tags) > 1:
                    print(
                        f"  ({len(pre_tags) - 1} older snapshot(s) available, use --history)"
                    )
                return

    # Fall back to .bak files
    backups = find_backups(path)
    if not backups:
        print(f"No backups found for {path}", file=sys.stderr)
        sys.exit(1)
    newest = backups[0]
    size_before = os.path.getsize(path) if os.path.exists(path) else 0
    size_backup = os.path.getsize(newest)
    shutil.copy2(newest, path)
    print(f"Restored from .bak: {newest}")
    print(
        f"  {size_backup / 1024 / 1024:.2f} MB (was {size_before / 1024 / 1024:.2f} MB)"
    )


def do_apply(input_path, reduced_path, profile_name="standard", cut=50, fade=75):
    """Replace original with reduced, using git for history and .bak as safety net."""
    if not os.path.exists(reduced_path):
        print(f"No reduced file found at {reduced_path}", file=sys.stderr)
        print(
            "Run without --apply first to generate the reduced file.", file=sys.stderr
        )
        sys.exit(1)

    # Safety: refuse if source file changed since the .reduced was generated.
    source_mtime = os.path.getmtime(input_path)
    reduced_mtime = os.path.getmtime(reduced_path)
    source_size = os.path.getsize(input_path)
    reduced_size = os.path.getsize(reduced_path)

    if source_mtime > reduced_mtime:
        print(
            "ERROR: Source file was modified after the .reduced file was generated.",
            file=sys.stderr,
        )
        print(
            f"  source:  {datetime.fromtimestamp(source_mtime):%Y-%m-%d %H:%M:%S} ({source_size / 1024 / 1024:.1f} MB)",
            file=sys.stderr,
        )
        print(
            f"  reduced: {datetime.fromtimestamp(reduced_mtime):%Y-%m-%d %H:%M:%S} ({reduced_size / 1024 / 1024:.1f} MB)",
            file=sys.stderr,
        )
        print(
            "The session may have new messages since reduction. Re-run without --apply first.",
            file=sys.stderr,
        )
        sys.exit(1)

    project_dir = os.path.dirname(os.path.abspath(input_path))
    basename = os.path.basename(input_path)
    short = session_short_id(input_path)

    # Git: snapshot pre-reduction state
    newly_init = ensure_git_repo(project_dir)
    if newly_init:
        print(f"Initialized git repo in {project_dir}")

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

    print(f"Applied: {input_path}")
    print(f"  {orig_size / 1024 / 1024:.2f} MB -> {new_size / 1024 / 1024:.2f} MB")
    print(f"  git tags: {pre_tag} -> {post_tag}")
    print(f"  .bak: {bak_path}")
    print(f"  restore: {sys.argv[0]} --restore {input_path}")


def do_history(path):
    """Show reduction timeline for a session."""
    project_dir = os.path.dirname(os.path.abspath(path))
    basename = os.path.basename(path)
    short = session_short_id(path)

    if not os.path.isdir(os.path.join(project_dir, ".git")):
        # Fall back to listing .bak files
        backups = find_backups(path)
        if not backups:
            print("No history. Run --apply to start tracking.")
            return
        print(f"Backup files for {short}... (no git history):\n")
        for bak in backups:
            size = os.path.getsize(bak)
            mtime = datetime.fromtimestamp(os.path.getmtime(bak))
            print(
                f"  {mtime:%Y-%m-%d %H:%M}  {size / 1024 / 1024:6.1f} MB  {os.path.basename(bak)}"
            )
        return

    tags = get_reduction_tags(project_dir, short)
    if not tags:
        print(f"No reductions recorded for {short}.")
        return

    # Group by timestamp
    reductions = {}
    for tag in tags:
        parts = tag.split("/")  # reduce/{short}/{ts}/{phase}
        if len(parts) == 4:
            ts, phase = parts[2], parts[3]
            reductions.setdefault(ts, {})[phase] = tag

    print(f"Reduction history for {short}:\n")
    for ts in sorted(reductions.keys()):
        phases = reductions[ts]
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
        print(f"  {ts_display}  {desc}")
        if pre_size is not None:
            print(f"    pre:  {pre_size / 1024 / 1024:6.1f} MB  ({pre_tag})")
        if post_size is not None:
            print(f"    post: {post_size / 1024 / 1024:6.1f} MB  ({post_tag})")
        if pre_size and post_size:
            saved = pre_size - post_size
            pct = saved / pre_size * 100
            print(f"    saved: {saved / 1024 / 1024:.1f} MB ({pct:.0f}%)")
        print()

    current_size = os.path.getsize(path) if os.path.exists(path) else 0
    print(
        f"{len(reductions)} reduction(s). Current: {current_size / 1024 / 1024:.1f} MB"
    )


def main():
    args = parse_args()
    INPUT = args.input
    OUTPUT = INPUT + ".reduced"

    if args.init:
        do_init(INPUT)
        return

    if args.restore:
        do_restore(INPUT)
        return

    if args.history:
        do_history(INPUT)
        return

    result = reduce_session(
        INPUT,
        profile=args.profile,
        cut=args.cut,
        fade=args.fade,
        chars_per_token=args.chars_per_token,
        estimate_tokens=args.tokens,
    )

    if not args.dry_run:
        with open(OUTPUT, "w") as f:
            f.writelines(result.kept_lines)

    saved = result.orig_size - result.new_size
    print(
        f"Original: {result.orig_count:,} lines, {result.orig_size / 1024 / 1024:.2f} MB"
    )
    print(
        f"Reduced:  {result.new_count:,} lines, {result.new_size / 1024 / 1024:.2f} MB"
    )
    print(
        f"Saved:    {result.orig_count - result.new_count:,} lines, {saved / 1024 / 1024:.2f} MB ({saved / result.orig_size * 100:.1f}%)"
    )
    print(f"Profile:  {args.profile}, cut={args.cut}%, fade={args.fade}%")
    print()
    for reason, count_val in sorted(result.stats.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count_val}")

    if result.orig_budget and result.reduced_budget:
        print(result.orig_budget.report(reduced_chars=result.reduced_budget._raw_chars))

    if args.dry_run:
        print("\n(dry run — no output written)")
    elif args.apply:
        print()
        do_apply(INPUT, OUTPUT, args.profile, args.cut, args.fade)
    else:
        print(f"\nOutput: {OUTPUT}")
        print(f"To apply:  {sys.argv[0]} --apply {INPUT}")
        print(f"To restore: {sys.argv[0]} --restore {INPUT}")
        print(f"To history: {sys.argv[0]} --history {INPUT}")


if __name__ == "__main__":
    main()
