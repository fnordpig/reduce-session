#!/usr/bin/env python3
"""Reduce a Claude Code session JSONL while preserving conversation quality.

Usage: reduce-session [options] <path-to-session.jsonl>

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
