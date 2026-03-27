"""MCP server for reduce-session — same functionality as the TUI, callable by Claude.

Thin wrappers over the existing session.py, doctor.py, reduction.py, and widgets.py
functions. No new logic — just serialization.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastmcp import FastMCP

from .session import (
    scan_projects,
    SessionInfo,
    resolve_slug_to_path,
    derive_project_name,
)
from .widgets import parse_browse_exchanges, BrowseExchange, get_section_snippet

mcp = FastMCP(
    "reduce-session",
    instructions=(
        "Inspect, diagnose, and reduce Claude Code session JSONL files. "
        "Start with list_sessions to see available sessions, then use "
        "browse_session to explore exchanges, doctor to diagnose issues, "
        "and reduce to compress."
    ),
)


def _get_projects_dir() -> Path:
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / "projects"
    return Path.home() / ".claude" / "projects"


def _session_to_dict(s: SessionInfo) -> dict:
    return {
        "project_name": s.project_name,
        "resolved_dir": str(s.resolved_dir) if s.resolved_dir else None,
        "session_id": s.session_id,
        "short_id": s.short_id,
        "path": str(s.path),
        "size_bytes": s.size_bytes,
        "size_mb": round(s.size_bytes / 1_000_000, 2),
        "token_estimate": s.token_estimate,
        "age": s.age_display,
        "line_count": s.line_count,
        "is_dangling": s.is_dangling,
        "continuation_count": len(s.continuation_files),
        "parse_error": s.parse_error,
    }


def _exchange_to_dict(ex: BrowseExchange) -> dict:
    return {
        "index": ex.index,
        "line": ex.index + 1,  # 1-indexed for display
        "role": ex.role,
        "text": ex.text,
        "tool_name": ex.tool_name,
        "is_error": ex.is_error,
        "ontology_class": ex.ontology_class,
        "reduce_route": ex.reduce_route,
        "token_size": ex.token_size,
        "is_structural": ex.is_structural,
        "is_distilled": ex.is_distilled,
        "output_file": ex.output_file,
        "output_file_exists": os.path.exists(ex.output_file)
        if ex.output_file
        else None,
    }


# ── Tools ────────────────────────────────────────────────────────────


@mcp.tool()
def list_sessions() -> str:
    """List all Claude Code sessions with metadata.

    Returns sessions grouped by project, sorted alphabetically.
    Each session includes: project name, resolved directory path,
    session ID, file size, estimated tokens, age, line count.
    """
    sessions = scan_projects(_get_projects_dir())
    projects: dict[str, list[dict]] = {}
    for s in sessions:
        projects.setdefault(s.project_name, []).append(_session_to_dict(s))

    result = {
        "total_sessions": len(sessions),
        "total_tokens": sum(s.token_estimate for s in sessions),
        "total_size_mb": round(sum(s.size_bytes for s in sessions) / 1_000_000, 2),
        "projects": projects,
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def browse_session(session_path: str, page: int = 0, prefix: str = "") -> str:
    """Browse session exchanges with hierarchical folding.

    Without prefix: returns top-level sections (up to 100 per page).
    With prefix like "§1-100": expands that section into sub-sections or leaves.

    Each section shows: range, token count, route distribution, snippet.
    Each leaf shows: index, role, ontology class, route, token size, text.

    Args:
        session_path: Full path to the .jsonl file
        page: Page number (0-indexed, 100 items per page)
        prefix: Section prefix to expand (e.g., "§1-100")
    """
    exchanges = parse_browse_exchanges(session_path)
    if not exchanges:
        return json.dumps({"error": "No exchanges found", "path": session_path})

    n = len(exchanges)

    # Parse prefix to get start/end range
    if prefix and prefix.startswith("§"):
        try:
            parts = prefix[1:].split("-")
            start = int(parts[0]) - 1  # convert from 1-indexed
            end = int(parts[1]) if len(parts) > 1 else start + 1
            end = min(end, n)
            start = max(start, 0)
        except (ValueError, IndexError):
            return json.dumps({"error": f"Invalid prefix: {prefix}"})
    else:
        start = 0
        end = n

    count = end - start

    if count <= 50:
        # Leaf level: return individual exchanges
        page_start = start + page * 100
        page_end = min(page_start + 100, end)
        leaves = [_exchange_to_dict(ex) for ex in exchanges[page_start:page_end]]
        return json.dumps(
            {
                "level": "leaf",
                "range": f"§{start + 1}-{end}",
                "total_exchanges": count,
                "page": page,
                "items": leaves,
            },
            indent=2,
        )

    # Section level: divide into chunks
    chunk_size = max(50, count // 100)
    sections = []
    for cs in range(start, end, chunk_size):
        ce = min(cs + chunk_size, end)
        section_exs = exchanges[cs:ce]
        total_tokens = sum(ex.token_size for ex in section_exs)
        snippet = get_section_snippet(section_exs)

        # Route distribution
        routes: dict[str, int] = {}
        for ex in section_exs:
            r = ex.reduce_route or "none"
            routes[r] = routes.get(r, 0) + 1

        # Ontology class distribution (top 3)
        classes: dict[str, int] = {}
        for ex in section_exs:
            if ex.ontology_class:
                classes[ex.ontology_class] = classes.get(ex.ontology_class, 0) + 1
        top_classes = sorted(classes.items(), key=lambda x: -x[1])[:3]

        sections.append(
            {
                "prefix": f"§{cs + 1}-{ce}",
                "count": ce - cs,
                "token_count": total_tokens,
                "token_display": _format_tokens(total_tokens),
                "routes": routes,
                "top_classes": [{"class": c, "count": n} for c, n in top_classes],
                "snippet": snippet,
            }
        )

    # Paginate sections
    page_start = page * 100
    page_end = min(page_start + 100, len(sections))

    return json.dumps(
        {
            "level": "sections",
            "range": f"§{start + 1}-{end}",
            "total_exchanges": count,
            "total_sections": len(sections),
            "page": page,
            "items": sections[page_start:page_end],
        },
        indent=2,
    )


@mcp.tool()
def get_exchange(session_path: str, index: int) -> str:
    """Get full content and metadata of a single exchange.

    Args:
        session_path: Full path to the .jsonl file
        index: 0-based line index in the JSONL file
    """
    exchanges = parse_browse_exchanges(session_path)
    for ex in exchanges:
        if ex.index == index:
            result = _exchange_to_dict(ex)
            result["full_text"] = ex.full_text
            if ex.output_file:
                result["output_file_size"] = (
                    os.path.getsize(ex.output_file)
                    if os.path.exists(ex.output_file)
                    else None
                )
            return json.dumps(result, indent=2)

    return json.dumps({"error": f"Exchange at index {index} not found"})


@mcp.tool()
def doctor(session_path: str) -> str:
    """Run all diagnostics on a session file.

    Returns 8 diagnostic results: compaction summaries, parent chain,
    stale tokens, overlapping files, unreduced metadata, reduce tags,
    bloated TUR, orphaned tool results.

    Each diagnostic has: name, severity (ok/info/warning/critical),
    summary, fix_description (if fixable).
    """
    from .doctor import (
        diagnose_compaction_summaries,
        diagnose_parent_chain,
        diagnose_stale_tokens,
        diagnose_overlapping_files,
        diagnose_unreduced_metadata,
        diagnose_reduce_tags,
        diagnose_bloated_tur,
        diagnose_orphaned_tool_results,
    )

    with open(session_path) as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    diagnostics = [
        diagnose_compaction_summaries(lines, session_path),
        diagnose_parent_chain(lines, session_path),
        diagnose_stale_tokens(lines, session_path),
        diagnose_overlapping_files(lines, session_path),
        diagnose_unreduced_metadata(lines, session_path),
        diagnose_reduce_tags(lines, session_path),
        diagnose_bloated_tur(lines, session_path),
        diagnose_orphaned_tool_results(lines, session_path),
    ]

    results = []
    for d in diagnostics:
        results.append(
            {
                "name": d.name,
                "severity": d.severity,
                "summary": d.summary,
                "fix_description": d.fix_description or None,
                "has_fix": d.fix_fn is not None,
                "detail": d.detail_lines,
            }
        )

    return json.dumps({"diagnostics": results}, indent=2)


@mcp.tool()
def doctor_fix(session_path: str, diagnostic_names: list[str]) -> str:
    """Apply selected doctor fixes to a session file.

    Git snapshots are created before and after fixes.

    Args:
        session_path: Full path to the .jsonl file
        diagnostic_names: List of diagnostic names to fix (e.g., ["compaction_summaries", "parent_chain"])
    """
    from .doctor import (
        diagnose_compaction_summaries,
        diagnose_parent_chain,
        diagnose_stale_tokens,
        diagnose_overlapping_files,
        diagnose_unreduced_metadata,
        diagnose_reduce_tags,
        diagnose_bloated_tur,
        diagnose_orphaned_tool_results,
        apply_fixes,
    )
    from .git_ops import ensure_git_repo, git_snapshot

    with open(session_path) as f:
        lines = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    p = Path(session_path)
    project_dir = str(p.parent)
    basename = p.name
    short = p.stem[:8]

    # Pre-fix snapshot
    try:
        ensure_git_repo(project_dir)
        git_snapshot(project_dir, basename, None, f"mcp doctor: pre-fix {short}")
    except Exception:
        pass

    # Run diagnostics and select the ones requested
    all_diags = [
        diagnose_compaction_summaries(lines, session_path),
        diagnose_parent_chain(lines, session_path),
        diagnose_stale_tokens(lines, session_path),
        diagnose_overlapping_files(lines, session_path),
        diagnose_unreduced_metadata(lines, session_path),
        diagnose_reduce_tags(lines, session_path),
        diagnose_bloated_tur(lines, session_path),
        diagnose_orphaned_tool_results(lines, session_path),
    ]

    selected = [d for d in all_diags if d.name in diagnostic_names and d.fix_fn]
    if not selected:
        return json.dumps(
            {"error": "No fixable diagnostics matched", "requested": diagnostic_names}
        )

    stats = apply_fixes(lines, session_path, selected)

    # Atomic write
    tmp_fd, tmp_path = tempfile.mkstemp(dir=project_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp_path, session_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    # Post-fix snapshot
    try:
        git_snapshot(
            project_dir,
            basename,
            None,
            f"mcp doctor: {', '.join(diagnostic_names)} ({short})",
        )
    except Exception:
        pass

    return json.dumps({"fixed": [d.name for d in selected], "stats": stats}, indent=2)


@mcp.tool()
def reduce(
    session_path: str,
    profile: str = "standard",
    llm: str | None = None,
    cut: float = 0.1,
    fade: float = 0.3,
) -> str:
    """Run the full reduction pipeline on a session file.

    Args:
        session_path: Full path to the .jsonl file
        profile: Reduction profile — "standard" or "aggressive"
        llm: LLM provider spec for classification (e.g., "anthropic", "ollama:llama3.2")
        cut: Head-zone cutoff fraction (default 0.1)
        fade: Fade-in fraction after cutoff (default 0.3)
    """
    from .reduction import reduce_session
    from .git_ops import do_apply

    llm_provider = None
    if llm:
        from .llm import create_provider

        llm_provider = create_provider(llm)

    result = reduce_session(
        session_path,
        cut=cut,
        fade=fade,
        profile=profile,
        llm_provider=llm_provider,
    )

    # Write the result
    reduced_path = session_path + ".reduced"
    with open(reduced_path, "w") as f:
        f.writelines(result.kept_lines)

    apply_result = do_apply(session_path, reduced_path, profile_name=profile)

    return json.dumps(
        {
            "orig_count": result.orig_count,
            "new_count": result.new_count,
            "orig_size": result.orig_size,
            "new_size": result.new_size,
            "reduction_pct": round(
                (1 - result.new_size / max(result.orig_size, 1)) * 100, 1
            ),
            "stats": result.stats,
            "apply": {
                "orig_size": apply_result.orig_size,
                "new_size": apply_result.new_size,
                "bak_path": apply_result.bak_path,
                "pre_tag": apply_result.pre_tag,
                "post_tag": apply_result.post_tag,
            },
        },
        indent=2,
    )


@mcp.tool()
def delete_exchange(session_path: str, index: int) -> str:
    """Delete a single exchange from a session file.

    Repairs the parentUuid chain and creates git snapshots.

    Args:
        session_path: Full path to the .jsonl file
        index: 0-based line index to delete
    """
    from .git_ops import ensure_git_repo, git_snapshot

    p = Path(session_path)
    project_dir = str(p.parent)

    # Pre-delete snapshot
    try:
        ensure_git_repo(project_dir)
        git_snapshot(project_dir, p.name, None, f"mcp: pre-delete line {index}")
    except Exception:
        pass

    with open(session_path) as f:
        lines = f.readlines()

    if index < 0 or index >= len(lines):
        return json.dumps({"error": f"Index {index} out of range (0-{len(lines) - 1})"})

    # Parse the line to delete for reparenting
    try:
        deleted_obj = json.loads(lines[index])
        deleted_uuid = deleted_obj.get("uuid")
        deleted_parent = deleted_obj.get("parentUuid")
    except (json.JSONDecodeError, AttributeError):
        deleted_uuid = None
        deleted_parent = None

    del lines[index]

    # Reparent children
    reparented = 0
    if deleted_uuid:
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
                if obj.get("parentUuid") == deleted_uuid:
                    obj["parentUuid"] = deleted_parent
                    lines[i] = json.dumps(obj, ensure_ascii=False) + "\n"
                    reparented += 1
            except json.JSONDecodeError:
                continue

    # Atomic write
    tmp_fd, tmp_path = tempfile.mkstemp(dir=project_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.writelines(lines)
        os.replace(tmp_path, session_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    # Post-delete snapshot
    try:
        git_snapshot(project_dir, p.name, None, f"mcp: deleted line {index}")
    except Exception:
        pass

    return json.dumps(
        {
            "deleted_index": index,
            "deleted_uuid": deleted_uuid,
            "reparented": reparented,
            "remaining_lines": len(lines),
        },
        indent=2,
    )


@mcp.tool()
def classify_exchange(
    session_path: str,
    index: int,
    profile: str = "standard",
) -> str:
    """Classify and compress a single exchange via LLM.

    Stamps _reduce tag with ontology class and route, then applies
    structural compression based on profile.

    Args:
        session_path: Full path to the .jsonl file
        index: 0-based line index to classify
        profile: "standard" (aggr=0.5) or "aggressive" (aggr=0.8)
    """
    from .reduction import (
        structural_compress,
        truncate,
        blended_limit,
        PROFILES,
        stamp_reduce_tag,
        get_reduce_tag,
    )
    from .git_ops import ensure_git_repo, git_snapshot

    with open(session_path) as f:
        raw_lines = f.readlines()

    if index < 0 or index >= len(raw_lines):
        return json.dumps({"error": f"Index {index} out of range"})

    obj = json.loads(raw_lines[index])

    # Ensure a reduce tag exists
    if not get_reduce_tag(obj):
        stamp_reduce_tag(obj, structural=True, profile=profile)

    # Apply structural compression
    prof = PROFILES.get(profile, PROFILES["standard"])
    agg_lim = prof["aggressive"]
    gen_lim = prof["gentle"]
    aggr = 0.5 if profile == "standard" else 0.8

    msg = obj.get("message", {})
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = structural_compress(content, aggr)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                for k in ("text", "content", "thinking"):
                    v = block.get(k, "")
                    if isinstance(v, str) and v:
                        compressed = structural_compress(v, aggr)
                        limit = blended_limit("default", aggr, agg_lim, gen_lim)
                        if len(compressed) > limit:
                            compressed = truncate(compressed, limit, k)
                        block[k] = compressed

    raw_lines[index] = json.dumps(obj, ensure_ascii=False) + "\n"

    # Atomic write
    p = Path(session_path)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.writelines(raw_lines)
        os.replace(tmp_path, session_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    try:
        ensure_git_repo(str(p.parent))
        git_snapshot(
            str(p.parent), p.name, None, f"mcp: classified line {index} ({profile})"
        )
    except Exception:
        pass

    tag = get_reduce_tag(obj)
    return json.dumps(
        {
            "index": index,
            "reduce_tag": tag,
            "profile": profile,
        },
        indent=2,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _format_tokens(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    if tokens >= 1_000:
        return f"{tokens // 1_000}k"
    return str(tokens)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
