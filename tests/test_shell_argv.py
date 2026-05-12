"""Tests for the shell-command-string → verb classifier.

The classifier consumes a raw command string (as Codex emits it inside
``function_call.arguments``) and returns a structured verb intent. It must
recover file-level semantics from common shell utilities without misclassifying
ambiguous cases.

Coverage strategy:
- One test per utility class (read / write / edit / build / generic).
- Edge cases: pipelines, redirections, multiple targets, shell wrappers
  (``sudo``, ``env``, ``bash -c``).
- Failure modes: lexer failures, empty strings, non-file argv.
"""

from __future__ import annotations

from reduce_session.shell_argv import ShellIntent, classify_shell_command


# ---------- Reads ----------

def test_cat_single_file_is_read():
    intent = classify_shell_command("cat /tmp/foo.py")
    assert intent.kind == "read"
    assert intent.paths == ("/tmp/foo.py",)


def test_cat_multiple_files_returns_all_paths():
    intent = classify_shell_command("cat a.py b.py c.py")
    assert intent.kind == "read"
    assert intent.paths == ("a.py", "b.py", "c.py")


def test_head_tail_sed_n_are_reads():
    for cmd in ("head -n 50 x.py", "tail -50 x.py", "sed -n '1,80p' x.py"):
        intent = classify_shell_command(cmd)
        assert intent.kind == "read", f"{cmd} → {intent.kind}"
        assert intent.paths == ("x.py",)


def test_rg_grep_with_path_are_reads():
    intent = classify_shell_command("rg 'pattern' src/main.py")
    assert intent.kind == "read"
    assert "src/main.py" in intent.paths


# ---------- Writes ----------

def test_redirect_to_file_is_write():
    intent = classify_shell_command("echo hello > /tmp/out.txt")
    assert intent.kind == "write"
    assert intent.paths == ("/tmp/out.txt",)


def test_append_redirect_is_write():
    intent = classify_shell_command("printf 'x\\n' >> log.txt")
    assert intent.kind == "write"
    assert intent.paths == ("log.txt",)


def test_tee_is_write():
    intent = classify_shell_command("cat data | tee /tmp/out")
    # Pipelines map to the dominant write target.
    assert intent.kind == "write"
    assert intent.paths == ("/tmp/out",)


def test_touch_is_write():
    intent = classify_shell_command("touch newfile.txt")
    assert intent.kind == "write"
    assert intent.paths == ("newfile.txt",)


# ---------- Edits ----------

def test_sed_in_place_is_edit():
    intent = classify_shell_command("sed -i 's/foo/bar/g' src/main.py")
    assert intent.kind == "edit"
    assert intent.paths == ("src/main.py",)


def test_apply_patch_is_edit():
    intent = classify_shell_command("apply_patch < /tmp/p.diff")
    assert intent.kind == "edit"


def test_patch_is_edit():
    intent = classify_shell_command("patch -p1 < /tmp/p.diff")
    assert intent.kind == "edit"


def test_git_apply_is_edit():
    intent = classify_shell_command("git apply /tmp/p.diff")
    assert intent.kind == "edit"


# ---------- Builds ----------

def test_pytest_is_build():
    intent = classify_shell_command("pytest tests/")
    assert intent.kind == "build"


def test_cargo_test_is_build():
    intent = classify_shell_command("cargo test --release")
    assert intent.kind == "build"


def test_npm_test_is_build():
    intent = classify_shell_command("npm test")
    assert intent.kind == "build"


def test_make_is_build():
    intent = classify_shell_command("make -j8 release")
    assert intent.kind == "build"


def test_python_dash_m_pytest_is_build():
    intent = classify_shell_command("python -m pytest -xvs")
    assert intent.kind == "build"


# ---------- Wrappers ----------

def test_sudo_unwraps_to_inner_verb():
    intent = classify_shell_command("sudo cat /etc/hosts")
    assert intent.kind == "read"
    assert intent.paths == ("/etc/hosts",)


def test_env_unwraps_to_inner_verb():
    intent = classify_shell_command("env FOO=bar pytest tests/")
    assert intent.kind == "build"


def test_bash_dash_c_unwraps_inner():
    intent = classify_shell_command("bash -c 'cat foo.py'")
    assert intent.kind == "read"
    assert intent.paths == ("foo.py",)


def test_time_prefix_unwraps():
    intent = classify_shell_command("time pytest tests/")
    assert intent.kind == "build"


# ---------- Fallbacks ----------

def test_unknown_utility_is_generic_run():
    intent = classify_shell_command("docker compose up -d")
    assert intent.kind == "run"
    assert intent.paths == ()


def test_empty_command_is_generic():
    intent = classify_shell_command("")
    assert intent.kind == "run"


def test_invalid_quoting_falls_back_to_run():
    """An unterminated quote should not crash; degrade to ``run``."""
    intent = classify_shell_command("echo 'unterminated")
    assert intent.kind == "run"


def test_pipeline_keeps_all_phase_paths():
    """``cat a | tee b`` should expose both the read and the write target
    so a downstream caller can decide to emit two events from one record."""
    intent = classify_shell_command("cat a.py | tee b.py")
    assert intent.kind == "write"
    assert "b.py" in intent.paths
    assert "a.py" in intent.read_through_paths


def test_sequence_and_and_picks_dominant_mutation():
    """``cat foo && rm foo`` must report the destructive ``rm``, not the
    cosmetic ``cat`` — silently absorbing ``&&`` as a path was the original
    bug."""
    intent = classify_shell_command("cat foo && rm foo")
    assert intent.kind == "write"  # rm classifies as write
    assert intent.paths == ("foo",)
    assert "foo" in intent.read_through_paths  # cat is preserved


def test_sequence_or_or_handled():
    intent = classify_shell_command("test -f x.py || touch x.py")
    assert intent.kind == "write"
    assert intent.paths == ("x.py",)


def test_sequence_semicolon_handled():
    intent = classify_shell_command("mkdir -p dist; touch dist/out")
    assert intent.kind == "write"
    # mkdir and touch both write; the dominant (rank) is whichever the
    # rank table picks — both are "write". The point is no operator absorbed.
    assert "&&" not in intent.paths
    assert ";" not in intent.paths


def test_sequence_with_pipeline_inside_clause():
    """Sequences nest pipelines: ``cd foo && cat a | tee b`` should classify
    the second clause as a pipeline write to b."""
    intent = classify_shell_command("cd foo && cat a | tee b")
    assert intent.kind == "write"
    assert intent.paths == ("b",)
    assert "a" in intent.read_through_paths


def test_sequence_all_generic_collapses_to_run():
    intent = classify_shell_command("echo start && date && whoami")
    assert intent.kind == "run"
    assert intent.paths == ()


def test_sequence_operators_not_treated_as_paths():
    """Regression: sequence operators must never appear in extracted paths."""
    for cmd in (
        "git status && git diff",
        "make clean ; make all",
        "ls foo || touch foo",
        "cmd1 & cmd2",
    ):
        intent = classify_shell_command(cmd)
        assert "&&" not in intent.paths
        assert "||" not in intent.paths
        assert ";" not in intent.paths
        assert "&" not in intent.paths


def test_intent_is_immutable():
    """Intent is a value object — downstream code should not mutate it."""
    intent = classify_shell_command("cat foo.py")
    assert isinstance(intent, ShellIntent)
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.kind = "write"  # type: ignore[misc]
