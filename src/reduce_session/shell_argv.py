"""Classify a raw shell command string into a verb intent.

Codex emits shell invocations as opaque command strings in
``function_call.arguments`` (or as a JSON dict containing one). To make the
reduction heuristics fire on Codex sessions, we have to recover file-level
semantics from those strings: which utility, which paths, which redirection.

This is intentionally a lossy classifier. It handles the common-and-clean
cases (single utility, simple redirection, well-known wrappers) and degrades
gracefully — anything ambiguous becomes a generic ``run`` intent rather than
a guess.

Pipelines are special: ``cat a | tee b`` is a read of ``a`` followed by a
write of ``b``. The classifier returns the *dominant* terminal phase
(here, the write) plus a ``read_through_paths`` field carrying the upstream
reads. Callers that care about both can emit two events.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Literal

ShellKind = Literal["read", "write", "edit", "build", "run"]

# Utilities whose argv consists of file targets (after option flags). These
# all classify as ``read`` and we extract the trailing positional args as
# paths.
_READ_UTILS: frozenset[str] = frozenset({
    "cat", "bat", "head", "tail", "less", "more", "nl", "od", "xxd",
    "ls", "stat", "file", "wc",
    "rg", "grep", "ag", "ack", "fgrep", "egrep",
    "find", "fd", "locate",
    "diff",  # diff a b — a and b are both read
})

# A few read utilities take a pattern as the first positional; the actual
# file paths start at the second positional onward.
_PATTERN_THEN_PATH: frozenset[str] = frozenset({
    "rg", "grep", "fgrep", "egrep", "ag", "ack",
})

# Sed is special: ``sed -i`` is edit-in-place, ``sed -n`` / ``sed -e`` are reads.
_SED = "sed"

# Utilities that write the file given as their last positional, with no input.
_WRITE_UTILS: frozenset[str] = frozenset({"touch", "mkdir", "cp", "mv", "ln", "rm"})

# Utilities that take patches as input.
_PATCH_UTILS: frozenset[str] = frozenset({"apply_patch", "patch"})

# Build/test runners — their output encodes pass/fail (``N passed`` /
# ``FAILED``) so detect_passing_builds can reason about it.
_BUILD_PROGRAMS: frozenset[str] = frozenset({
    "pytest", "py.test", "unittest",
    "cargo", "rustc",
    "go", "gotest",
    "npm", "yarn", "pnpm", "bun",
    "make", "cmake", "ninja",
    "tox", "nox",
    "mvn", "gradle", "sbt",
    "ctest",
    "bazel", "buck",
    "tsc", "swc",
})

# Linters / formatters / type-checkers — these also "pass or fail" but with
# different output conventions (no ``N passed`` aggregate). Kept distinct so
# build-specific heuristics don't over-trigger on lint output. For now both
# sets classify as ``build``, but a future split can promote linters to
# their own verb without touching the codec layer.
_LINTER_PROGRAMS: frozenset[str] = frozenset({
    "ruff", "mypy", "ty", "pyright", "black", "flake8", "eslint",
    "prettier", "isort", "autopep8",
    "uv", "pip", "poetry",
})

# Argv prefixes that we strip and re-classify against the rest.
_WRAPPERS: frozenset[str] = frozenset({
    "sudo", "doas", "env", "nohup", "ionice", "nice", "time", "timeout",
    "stdbuf", "command", "exec", "builtin",
    "xargs",  # technically wraps inner cmd
})


@dataclass(frozen=True, slots=True)
class ShellIntent:
    """Verb-level interpretation of a shell command string."""

    kind: ShellKind
    paths: tuple[str, ...] = ()
    read_through_paths: tuple[str, ...] = field(default_factory=tuple)
    program: str | None = None


def _safe_split(cmd: str) -> list[str] | None:
    """Tokenize as POSIX shell, returning None on lexer failure."""
    try:
        return shlex.split(cmd, posix=True, comments=False)
    except ValueError:
        return None


def _strip_redirection(argv: list[str]) -> tuple[list[str], list[str], bool]:
    """Pull out ``>``, ``>>``, and stdin-redirect targets.

    Returns (argv_without_redir, write_targets, has_stdin_redirect)."""
    write_targets: list[str] = []
    has_stdin_redirect = False
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in (">", ">>", "1>", "1>>", "2>", "2>>", "&>", "&>>"):
            if i + 1 < len(argv):
                write_targets.append(argv[i + 1])
                i += 2
                continue
        if tok == "<" or tok.startswith("<"):
            has_stdin_redirect = True
            i += 1
            continue
        out.append(tok)
        i += 1
    return out, write_targets, has_stdin_redirect


def _unwrap_prefix(argv: list[str]) -> list[str]:
    """Strip leading wrappers (``sudo``, ``env``, ``time``, ``bash -c``) so
    the program at argv[0] is the real verb. ``env`` skips ``KEY=VAL`` args."""
    while argv:
        head = argv[0]
        if head in {"bash", "sh", "zsh"} and len(argv) >= 3 and argv[1] == "-c":
            inner = _safe_split(argv[2])
            if inner is None:
                return argv  # give up
            argv = inner
            continue
        if head not in _WRAPPERS:
            return argv
        argv = argv[1:]
        if head == "env":
            while argv and "=" in argv[0] and not argv[0].startswith("-"):
                argv = argv[1:]
        if head == "timeout" and argv:
            argv = argv[1:]  # skip duration
        if head == "xargs":
            while argv and argv[0].startswith("-"):
                argv = argv[1:]
                # Some flags take an arg; for our purposes we don't need to
                # be exact — the inner program is what matters.
    return argv


def _positionals(argv: list[str]) -> list[str]:
    """Return non-flag args (everything not starting with ``-``).

    Numeric tokens are skipped because they are almost always flag arguments
    in the utilities we care about (``head -n 50``, ``tail -50``)."""
    out: list[str] = []
    for a in argv[1:]:
        if a.startswith("-"):
            continue
        if a.isdigit():
            continue
        out.append(a)
    return out


def _classify_single(argv: list[str]) -> ShellIntent:
    argv = _unwrap_prefix(argv)
    if not argv:
        return ShellIntent(kind="run")
    argv, write_targets, has_stdin = _strip_redirection(argv)
    if not argv:
        if write_targets:
            return ShellIntent(kind="write", paths=tuple(write_targets))
        return ShellIntent(kind="run")
    prog = argv[0]
    positionals = _positionals(argv)

    # Explicit redirection trumps program defaults.
    if write_targets:
        return ShellIntent(
            kind="write",
            paths=tuple(write_targets),
            read_through_paths=tuple(positionals) if prog in _READ_UTILS else (),
            program=prog,
        )

    if prog == _SED:
        # sed -i  → edit (in-place)
        # sed -n  → read
        # plain sed with no -i → treat as read on input file (best effort).
        # The first positional is the sed expression; file paths follow.
        sed_paths = tuple(positionals[1:]) if positionals else ()
        if "-i" in argv or any(a.startswith("-i") and a != "--include" for a in argv):
            return ShellIntent(kind="edit", paths=sed_paths, program=prog)
        return ShellIntent(kind="read", paths=sed_paths, program=prog)

    if prog in _PATCH_UTILS:
        # Patches edit existing files. Don't try to extract paths from the
        # patch text — those are inside the diff blob, not argv.
        return ShellIntent(kind="edit", program=prog)

    if prog == "git" and len(argv) >= 2 and argv[1] == "apply":
        return ShellIntent(kind="edit", program=prog)

    if prog in _BUILD_PROGRAMS or prog in _LINTER_PROGRAMS:
        return ShellIntent(kind="build", program=prog)

    if prog == "python" or prog.startswith("python"):
        # python -m pytest, python -m unittest, etc.
        if any(a in _BUILD_PROGRAMS for a in argv[2:3]):
            return ShellIntent(kind="build", program=prog)
        return ShellIntent(kind="run", program=prog)

    if prog == "tee":
        return ShellIntent(kind="write", paths=tuple(positionals), program=prog)

    if prog in _WRITE_UTILS:
        return ShellIntent(kind="write", paths=tuple(positionals), program=prog)

    if prog in _READ_UTILS:
        paths: tuple[str, ...]
        if prog in _PATTERN_THEN_PATH:
            # First positional is the pattern; paths follow. With only one
            # positional (just the pattern), nothing was read by name — the
            # command read from stdin / pipeline / current directory. Don't
            # mis-classify the pattern as a file path.
            paths = tuple(positionals[1:]) if len(positionals) >= 2 else ()
        else:
            paths = tuple(positionals)
        return ShellIntent(kind="read", paths=paths, program=prog)

    return ShellIntent(kind="run", program=prog)


_PIPELINE_SEPARATORS: frozenset[str] = frozenset({"|"})
_SEQUENCE_SEPARATORS: frozenset[str] = frozenset({"&&", "||", ";", "&"})


def _split_pipeline(argv: list[str]) -> list[list[str]]:
    """Split argv on ``|`` tokens. Quoted pipes are absorbed by shlex
    into a single token, so a literal ``|`` here is always a pipeline boundary."""
    out: list[list[str]] = []
    current: list[str] = []
    for tok in argv:
        if tok in _PIPELINE_SEPARATORS:
            if current:
                out.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        out.append(current)
    return out


def _split_sequences(argv: list[str]) -> list[list[str]]:
    """Split argv on shell sequence operators (``&&``, ``||``, ``;``, ``&``).

    Real agent traces are full of ``mkdir -p foo && cd foo`` and the like;
    treating each clause as an independent command-or-pipeline keeps the
    classifier from silently absorbing operator tokens as bogus paths.
    """
    out: list[list[str]] = []
    current: list[str] = []
    for tok in argv:
        if tok in _SEQUENCE_SEPARATORS:
            if current:
                out.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        out.append(current)
    return out


def classify_shell_command(cmd: str) -> ShellIntent:
    """Classify a raw shell command string into a verb intent.

    Degrades to ``run`` for empty input, lexer failures, or unrecognized
    programs. Sequence operators (``&&``, ``||``, ``;``, ``&``) split into
    independent clauses; pipelines (``|``) split into phases within a
    clause. The returned intent prioritizes mutation: if any clause writes
    or edits, that's the verb; otherwise the first non-``run`` clause wins;
    otherwise ``run``. Upstream reads (from pipelines, or from earlier
    clauses) are collected into ``read_through_paths``."""
    if not cmd or not cmd.strip():
        return ShellIntent(kind="run")
    argv = _safe_split(cmd)
    if argv is None or not argv:
        return ShellIntent(kind="run")

    clause_intents: list[ShellIntent] = []
    for clause in _split_sequences(argv):
        if not clause:
            continue
        clause_intents.append(_classify_pipeline(clause))
    if not clause_intents:
        return ShellIntent(kind="run")
    if len(clause_intents) == 1:
        return clause_intents[0]

    return _merge_sequence_intents(clause_intents)


def _classify_pipeline(argv: list[str]) -> ShellIntent:
    phases = _split_pipeline(argv)
    intents = [_classify_single(p) for p in phases]
    if not intents:
        return ShellIntent(kind="run")
    if len(intents) == 1:
        return intents[0]
    terminal = intents[-1]
    upstream_reads: list[str] = []
    for inter in intents[:-1]:
        if inter.kind == "read":
            upstream_reads.extend(inter.paths)
    return ShellIntent(
        kind=terminal.kind,
        paths=terminal.paths,
        read_through_paths=tuple(upstream_reads),
        program=terminal.program,
    )


# Verbs by mutation strength — higher index = "more invasive". When a chain
# of clauses contains multiple verbs, the strongest wins so a sequence like
# ``cat foo && rm foo`` reports ``edit`` (rm), not ``read`` (cat).
_KIND_RANK: dict[str, int] = {"run": 0, "read": 1, "build": 2, "write": 3, "edit": 4}


def _merge_sequence_intents(intents: list[ShellIntent]) -> ShellIntent:
    dominant = max(intents, key=lambda i: _KIND_RANK.get(i.kind, 0))
    if _KIND_RANK.get(dominant.kind, 0) == 0:
        # All clauses are generic ``run`` — collapse to a single run intent.
        return ShellIntent(kind="run", program=intents[-1].program)
    upstream_reads: list[str] = []
    for inter in intents:
        if inter is dominant:
            continue
        if inter.kind == "read":
            upstream_reads.extend(inter.paths)
        upstream_reads.extend(inter.read_through_paths)
    merged_reads = tuple(dict.fromkeys(list(dominant.read_through_paths) + upstream_reads))
    return ShellIntent(
        kind=dominant.kind,
        paths=dominant.paths,
        read_through_paths=merged_reads,
        program=dominant.program,
    )
