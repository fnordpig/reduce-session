#!/usr/bin/env python3
"""Reduce a Claude Code session JSONL while preserving conversation quality.

Usage: reduce-session [options] <path-to-session.jsonl>

Options:
  --tokens           Print context-window token estimate by category
  --cut PCT          Start of gentle-to-aggressive ramp (default: 10)
  --fade PCT         Start of aggressive-to-gentle ramp (default: 75)
  --profile NAME     Preset: gentle, standard (default), aggressive
  --dry-run          Analyze only, don't write output
  --apply            Replace original with reduced (backs up to .bak first)
  --restore          Restore from most recent .bak file
  --chars-per-token  Override chars/token ratio (default: 3.7)

Uses a U-curve gradient matching the "Lost in the Middle" LLM attention pattern:
gentle at start and end (high recall zones), aggressive in the middle.
"""

import argparse
import json
import os
import sys

from reduce_session.git_ops import (
    do_apply,
    do_history,
    do_init,
    do_restore,
)
from reduce_session.reduction import (
    CHARS_PER_TOKEN,
    reduce_session,
)


# ---------------------------------------------------------------------------
# Doctor subcommand
# ---------------------------------------------------------------------------

# Severity → terminal icon
_DOCTOR_ICONS = {
    "ok": "\u2713",  # ✓
    "warning": "\u26a0",  # ⚠
    "critical": "\u2717",  # ✗
    "info": "\u26a0",  # ⚠
}

# Exit-code contract:
#   0 = all ok / info only
#   1 = warnings (no critical)
#   2 = critical issues present
#   3 = parse failure
_EXIT_OK = 0
_EXIT_WARN = 1
_EXIT_CRITICAL = 2
_EXIT_PARSE_FAIL = 3


def _load_session_lines(session_path: str) -> list[dict]:
    """Parse a JSONL session file into a list of dicts."""
    lines: list[dict] = []
    with open(session_path, "r", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    lines.append(obj)
            except (json.JSONDecodeError, ValueError):
                pass
    return lines


def _run_all_checks(lines: list[dict], session_path: str):
    """Run all diagnostic checks and return results."""
    from reduce_session.doctor import (
        diagnose_bloated_tur,
        diagnose_compaction_summaries,
        diagnose_corrupted_content_blocks,
        diagnose_corrupted_tool_use,
        diagnose_cycle_in_parent_chain,
        diagnose_null_parentUuid_at_non_root,
        diagnose_orphaned_tool_results,
        diagnose_overlapping_files,
        diagnose_oversized_sessions,
        diagnose_parent_chain,
        diagnose_reduce_tags,
        diagnose_stale_backups,
        diagnose_stale_tokens,
        diagnose_unreduced_metadata,
    )

    checks = [
        diagnose_compaction_summaries,
        diagnose_corrupted_tool_use,
        diagnose_corrupted_content_blocks,
        diagnose_parent_chain,
        diagnose_cycle_in_parent_chain,
        diagnose_null_parentUuid_at_non_root,
        diagnose_stale_tokens,
        diagnose_overlapping_files,
        diagnose_unreduced_metadata,
        diagnose_reduce_tags,
        diagnose_bloated_tur,
        diagnose_orphaned_tool_results,
        diagnose_stale_backups,
        diagnose_oversized_sessions,
    ]
    return [check(lines, session_path) for check in checks]


def _print_doctor_results(results, fixed_names: set[str] | None = None) -> None:
    """Print one line per check: icon severity name summary [FIXED]."""
    if fixed_names is None:
        fixed_names = set()
    for r in results:
        icon = _DOCTOR_ICONS.get(r.severity, "?")
        fixed_marker = "  [FIXED]" if r.name in fixed_names else ""
        print(f"  {icon} [{r.severity:8s}] {r.name:36s} {r.summary}{fixed_marker}")


def _doctor_exit_code(results) -> int:
    severities = {r.severity for r in results}
    if "critical" in severities:
        return _EXIT_CRITICAL
    if "warning" in severities:
        return _EXIT_WARN
    return _EXIT_OK


def cmd_doctor(session_path: str, fix: bool) -> int:
    """Doctor entry point. Returns exit code."""
    # Parse
    try:
        lines = _load_session_lines(session_path)
    except OSError as e:
        print(f"Error: cannot read {session_path}: {e}", file=sys.stderr)
        return _EXIT_PARSE_FAIL
    except Exception as e:
        print(f"Error: failed to parse {session_path}: {e}", file=sys.stderr)
        return _EXIT_PARSE_FAIL

    print(f"Doctor: {session_path}")
    print()

    results = _run_all_checks(lines, session_path)

    if not fix:
        _print_doctor_results(results)
        print()
        return _doctor_exit_code(results)

    # --fix: run all fixable diagnostics in _FIX_ORDER priority
    from reduce_session.doctor import _FIX_ORDER, apply_fixes

    fixable = [r for r in results if r.fix_fn is not None]
    fixable_sorted = sorted(fixable, key=lambda r: _FIX_ORDER.get(r.name, 99))

    if not fixable_sorted:
        print("No fixable issues found.")
        _print_doctor_results(results)
        print()
        return _doctor_exit_code(results)

    print(f"Applying {len(fixable_sorted)} fix(es)...")
    apply_fixes(lines, session_path, fixable_sorted)

    # Write fixed lines back to file
    try:
        with open(session_path, "w") as fh:
            for obj in lines:
                fh.write(json.dumps(obj) + "\n")
    except OSError as e:
        print(f"Error: failed to write {session_path}: {e}", file=sys.stderr)
        return _EXIT_PARSE_FAIL

    fixed_names = {r.name for r in fixable_sorted}

    # Re-run checks on the now-fixed data to show updated state
    results_after = _run_all_checks(lines, session_path)
    _print_doctor_results(results_after, fixed_names)
    print()
    return _doctor_exit_code(results_after)


def parse_args():
    p = argparse.ArgumentParser(description="Reduce Claude Code session JSONL")
    p.add_argument("input", nargs="?", default=None, help="Path to session JSONL file")
    p.add_argument(
        "--browse",
        action="store_true",
        help="Launch interactive TUI session browser",
    )
    p.add_argument(
        "--tokens", action="store_true", help="Print token estimate by category"
    )
    p.add_argument(
        "--cut",
        type=int,
        default=10,
        help="Start of gentle-to-aggressive ramp, as %% of conversation (default: 10)",
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
    p.add_argument(
        "--llm",
        type=str,
        default=None,
        help="LLM provider (default: ollama if OLLAMA_HOST set, else local). "
        "Use 'none' to disable. "
        "Examples: local, ollama:qwen3:4b, anthropic:haiku, openai:gpt-4o-mini, gemini:flash. "
        "Env: REDUCE_SESSION_LLM",
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Run diagnostic checks on session file",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="With --doctor: apply all auto-fixable diagnostics in priority order",
    )
    return p.parse_args()


# --- Output formatting ---


def _print_init(result):
    """Format do_init result for terminal."""
    if not result.newly_created:
        print(f"Git repo already exists: {result.project_dir}")
        print(f"  {result.jsonl_count} session file(s)")
        print(f"  {result.tag_count} reduction tag(s)")
        if result.uncommitted_lines:
            print("  uncommitted changes:")
            for line in result.uncommitted_lines:
                print(f"    {line}")
    else:
        print(f"Initialized git repo: {result.project_dir}")
        print(f"  {result.jsonl_count} session file(s) tracked")
        print("  .gitignore configured (tracks *.jsonl only)")
        if result.initial_commit:
            print(f"  initial commit: {result.initial_commit}")


def _print_restore(result):
    """Format do_restore result for terminal."""
    if result.source == "git":
        print(f"Restored from git tag: {result.detail}")
        print(
            f"  {result.size_before / 1024 / 1024:.2f} MB -> {result.size_after / 1024 / 1024:.2f} MB"
        )
        if result.older_snapshots > 0:
            print(
                f"  ({result.older_snapshots} older snapshot(s) available, use --history)"
            )
    else:
        print(f"Restored from .bak: {result.detail}")
        print(
            f"  {result.size_after / 1024 / 1024:.2f} MB (was {result.size_before / 1024 / 1024:.2f} MB)"
        )


def _print_apply(result):
    """Format do_apply result for terminal."""
    if result.newly_init:
        project_dir = os.path.dirname(os.path.abspath(result.input_path))
        print(f"Initialized git repo in {project_dir}")
    print(f"Applied: {result.input_path}")
    print(
        f"  {result.orig_size / 1024 / 1024:.2f} MB -> {result.new_size / 1024 / 1024:.2f} MB"
    )
    print(f"  git tags: {result.pre_tag} -> {result.post_tag}")
    print(f"  .bak: {result.bak_path}")
    print(f"  restore: {sys.argv[0]} --restore {result.input_path}")


def _print_history(result):
    """Format do_history result for terminal."""
    if not result.has_git:
        if not result.backups:
            print("No history. Run --apply to start tracking.")
            return
        print(f"Backup files for {result.session_id}... (no git history):\n")
        for bak_path, size, mtime in result.backups:
            print(
                f"  {mtime:%Y-%m-%d %H:%M}  {size / 1024 / 1024:6.1f} MB  {os.path.basename(bak_path)}"
            )
        return

    if not result.reductions:
        print(f"No reductions recorded for {result.session_id}.")
        return

    print(f"Reduction history for {result.session_id}:\n")
    for entry in result.reductions:
        print(f"  {entry.ts_display}  {entry.description}")
        if entry.pre_size is not None:
            print(
                f"    pre:  {entry.pre_size / 1024 / 1024:6.1f} MB  ({entry.pre_tag})"
            )
        if entry.post_size is not None:
            print(
                f"    post: {entry.post_size / 1024 / 1024:6.1f} MB  ({entry.post_tag})"
            )
        if entry.saved_bytes is not None:
            print(
                f"    saved: {entry.saved_bytes / 1024 / 1024:.1f} MB ({entry.saved_pct:.0f}%)"
            )
        print()

    print(
        f"{len(result.reductions)} reduction(s). Current: {result.current_size / 1024 / 1024:.1f} MB"
    )


def main():
    args = parse_args()

    # Resolve LLM provider
    # Default: ollama if OLLAMA_HOST is set, otherwise local.
    # Only paid API providers (anthropic, openai, gemini) require explicit opt-in.
    llm_spec = args.llm or os.environ.get("REDUCE_SESSION_LLM")
    if llm_spec is None:
        if os.environ.get("OLLAMA_HOST"):
            llm_spec = "ollama"
        else:
            llm_spec = "local"
    if llm_spec == "none":
        llm_spec = None  # explicit opt-out

    llm_provider = None
    if llm_spec:
        try:
            from reduce_session.llm import create_provider

            llm_provider = create_provider(llm_spec)
        except Exception as e:
            print(f"Warning: LLM provider ({llm_spec}) failed: {e}", file=sys.stderr)
            print("Falling back to heuristic-only mode.", file=sys.stderr)

    # Doctor subcommand — dispatch before LLM init (no LLM needed)
    if args.doctor:
        if args.input is None:
            print(
                "Error: reduce-session --doctor requires a session file path.",
                file=sys.stderr,
            )
            sys.exit(_EXIT_PARSE_FAIL)
        code = cmd_doctor(args.input, fix=args.fix)
        sys.exit(code)

    # Launch TUI if --browse or no positional arg (and no action flags)
    if args.browse or (
        args.input is None and not any([args.restore, args.history, args.init])
    ):
        from .tui import SessionBrowserApp

        app = SessionBrowserApp(llm_spec=llm_spec)
        app.run()
        return

    if args.input is None:
        print(
            "Error: a session file path is required for this operation.",
            file=sys.stderr,
        )
        sys.exit(1)

    INPUT = args.input
    OUTPUT = INPUT + ".reduced"

    if args.init:
        try:
            result = do_init(INPUT)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        _print_init(result)
        return

    if args.restore:
        try:
            result = do_restore(INPUT)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        _print_restore(result)
        return

    if args.history:
        result = do_history(INPUT)
        _print_history(result)
        return

    result = reduce_session(
        INPUT,
        profile=args.profile,
        cut=args.cut,
        fade=args.fade,
        chars_per_token=args.chars_per_token,
        estimate_tokens=args.tokens,
        llm_provider=llm_provider,
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
        print("\n(dry run -- no output written)")
    elif args.apply:
        print()
        try:
            apply_result = do_apply(INPUT, OUTPUT, args.profile, args.cut, args.fade)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        _print_apply(apply_result)
    else:
        print(f"\nOutput: {OUTPUT}")
        print(f"To apply:  {sys.argv[0]} --apply {INPUT}")
        print(f"To restore: {sys.argv[0]} --restore {INPUT}")
        print(f"To history: {sys.argv[0]} --history {INPUT}")


if __name__ == "__main__":
    main()
