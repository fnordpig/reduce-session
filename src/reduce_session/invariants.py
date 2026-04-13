"""Safety invariants for session file writes.

Provides atomic writes, protection taxonomy, parent-chain relinking,
concurrent-access locking, and write-time conflict detection.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically: tmp → flush → fsync → os.replace."""
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write *text* to *path* atomically."""
    atomic_write_bytes(Path(path), text.encode(encoding))


def atomic_write_jsonl(
    path: Path, objs: list[dict], create_backup: bool = True
) -> None:
    """Write *objs* as JSONL to *path* atomically.

    If *create_backup* is True, the existing file (if any) is copied to a
    timestamped ``.bak`` file before the replace.
    """
    path = Path(path)
    if create_backup and path.exists():
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = path.parent / f"{path.name}.{ts}.bak"
        import shutil

        shutil.copy2(path, bak)

    lines = "".join(json.dumps(obj, ensure_ascii=False) + "\n" for obj in objs)
    atomic_write_text(path, lines)


# ---------------------------------------------------------------------------
# Protection taxonomy
# ---------------------------------------------------------------------------

_PROTECTED_TYPES = frozenset(
    {
        "content-replacement",
        "marble-origami-commit",
        "marble-origami-snapshot",
        "worktree-state",
        "task-summary",
    }
)

_BOUNDARY_SUBTYPES = frozenset({"compact_boundary", "microcompact_boundary"})


def is_protected(obj: dict) -> bool:
    """Return True when *obj* must not be dropped by any reduction pass."""
    # Explicit preservation marker set by reduce-session itself
    if obj.get("__reduce_session_preserved__"):
        return True

    # Visible-in-transcript-only items carry UI metadata
    if obj.get("isVisibleInTranscriptOnly"):
        return True

    t = obj.get("type")

    # Special top-level object types
    if t in _PROTECTED_TYPES:
        return True

    # User compact summary messages
    if t == "user" and obj.get("isCompactSummary"):
        return True

    # System compact/microcompact boundary records
    if t == "system":
        subtype = obj.get("subtype") or obj.get("message", {}).get("subtype")
        if subtype in _BOUNDARY_SUBTYPES:
            return True

    return False


# ---------------------------------------------------------------------------
# Parent-chain relinking
# ---------------------------------------------------------------------------


def relink_parent_chains(
    kept_objs: list[dict], dropped_uuids: dict[str, str | None]
) -> int:
    """Walk both ``parentUuid`` and ``logicalParentUuid`` through *dropped_uuids*.

    Mutates *kept_objs* in place.  Returns the number of fields relinked.

    Safety guarantees
    -----------------
    * Cycle guard via a ``visited`` set — each UUID is followed at most once
      per chain walk.
    * If a chain resolves to ``None`` or to an unknown UUID (not in
      *dropped_uuids*), the **original field value is preserved** — it is never
      overwritten with ``None``.
    """
    if not dropped_uuids:
        return 0

    relinked = 0

    for obj in kept_objs:
        for field_name in ("parentUuid", "logicalParentUuid"):
            original = obj.get(field_name)
            if original is None or original not in dropped_uuids:
                continue

            visited: set[str] = set()
            current: str | None = original
            while current is not None and current in dropped_uuids:
                if current in visited:
                    # Cycle detected — stop here; current is already in the
                    # dropped map so we cannot safely resolve further.
                    current = None
                    break
                visited.add(current)
                current = dropped_uuids[current]

            # Only update if we resolved to a real (non-None) UUID that is NOT
            # itself in the dropped set — i.e., it's a live ancestor.
            if current is not None and current not in dropped_uuids:
                obj[field_name] = current
                relinked += 1
            # else: chain exhausted or cycled → leave original value intact

    return relinked


# ---------------------------------------------------------------------------
# Concurrent-access lock
# ---------------------------------------------------------------------------


class PruneLockError(Exception):
    """Raised when a prune lock cannot be acquired (another process holds it)."""


@contextmanager
def prune_lock(session_path: Path):
    """Exclusive advisory lock around prune/reduce operations.

    Creates ``{session_path}.prune-lock`` and acquires ``LOCK_EX | LOCK_NB``.
    Raises :exc:`PruneLockError` immediately if the lock is already held.
    No-op on Windows (``fcntl`` is unavailable).
    """
    if sys.platform == "win32":
        yield
        return

    lock_path = Path(str(session_path) + ".prune-lock")
    fh = lock_path.open("w")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise PruneLockError(
                f"Another process is currently pruning {session_path}. Try again later."
            )
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()
        try:
            lock_path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Write-time conflict detection
# ---------------------------------------------------------------------------


class PruneConflictError(Exception):
    """Raised when a write-time conflict is detected (non-append changes)."""


def _md5_prefix(path: Path, size: int) -> str:
    """Return MD5 hex digest of the first *size* bytes of *path*."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        remaining = size
        while remaining > 0:
            chunk = fh.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


@dataclass
class FileSnapshot:
    """Captures file state at construction time for later conflict detection."""

    size: int
    mtime_ns: int
    md5: str
    _path: Path

    def __init__(self, path: Path) -> None:
        path = Path(path)
        self._path = path
        st = path.stat()
        self.size = st.st_size
        self.mtime_ns = st.st_mtime_ns
        h = hashlib.md5()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        self.md5 = h.hexdigest()

    def classify_against(
        self, path: Path
    ) -> Literal["unchanged", "appended", "conflict"]:
        """Compare snapshot against current file state.

        Returns
        -------
        ``"unchanged"``
            File is byte-for-byte identical (same size + same MD5).
        ``"appended"``
            File is larger and the first *snapshot.size* bytes still match
            (Claude Code appended new messages while we were reducing).
        ``"conflict"``
            Anything else — middle-of-file edits, truncation, etc.
        """
        path = Path(path)
        try:
            st = path.stat()
        except FileNotFoundError:
            return "conflict"

        current_size = st.st_size

        if current_size == self.size:
            # Same size — check full digest
            h = hashlib.md5()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return "unchanged" if h.hexdigest() == self.md5 else "conflict"

        if current_size > self.size:
            # Possibly appended — verify prefix
            prefix_md5 = _md5_prefix(path, self.size)
            return "appended" if prefix_md5 == self.md5 else "conflict"

        # Truncated
        return "conflict"


def write_jsonl_conflict_safe(
    path: Path,
    new_objs: list[dict],
    snapshot: FileSnapshot,
) -> None:
    """Write *new_objs* to *path*, handling concurrent Claude Code appends.

    Behaviour depends on :meth:`FileSnapshot.classify_against`:

    * ``"unchanged"`` — normal :func:`atomic_write_jsonl`.
    * ``"appended"`` — appended lines are read and concatenated to
      *new_objs* before writing, so no Claude Code messages are lost.
    * ``"conflict"`` — raises :exc:`PruneConflictError`.
    """
    path = Path(path)
    classification = snapshot.classify_against(path)

    if classification == "conflict":
        raise PruneConflictError(
            f"File {path} was modified in a way that cannot be safely merged. "
            "Re-run the reduction against the current file."
        )

    objs_to_write = list(new_objs)

    if classification == "appended":
        # Read only the newly appended lines (beyond snapshot.size bytes)
        with path.open("rb") as fh:
            fh.seek(snapshot.size)
            tail = fh.read().decode("utf-8", errors="replace")
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objs_to_write.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Skip malformed lines from partial writes

    atomic_write_jsonl(path, objs_to_write, create_backup=True)
