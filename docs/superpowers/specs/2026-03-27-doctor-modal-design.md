## Overview
A Doctor modal for the reduce-session TUI that diagnoses and fixes session health issues. Opened via `D` on a selected session. Runs 7 diagnostics with Tufte-style sparkline visualizations, presents fixable issues as a checklist with before/after previews, and applies selected fixes atomically with git-backed preservation.

## Diagnostics (ordered by severity)

### Critical
1. **Compaction summaries** — "being continued from a previous conversation" messages that create orphaned tree roots
2. **Broken parentUuid chains** — messages pointing to non-existent UUIDs

### Warning
3. **Stale token counter** — last message.usage doesn't match actual content

### Info
4. **Overlapping JSONL files** — continuation files that double-load context
5. **Unreduced metadata** — progress/file-history/queue-operation lines still present
6. **Missing _reduce tags** — middle-zone messages not yet LLM-processed
7. **Bloated toolUseResult** — originalFile/stdout fields still oversized

## DiagnosticResult dataclass
```python
@dataclass
class DiagnosticResult:
    name: str
    severity: str           # "critical", "warning", "ok", "info"
    summary: str
    sparkline_data: list    # position-aware visualization data
    fix_description: str    # preview of what fix does
    fix_fn: Callable | None # None = report only, not auto-fixable
    detail_lines: list[str]
```

## Layout
Full-width modal, single scrollable panel. Each diagnostic renders:
- Status icon + name + severity badge
- Sparkline visualization (type varies per diagnostic)
- Fix preview line with checkbox (fixable items only)

## Color scheme (Tufte-inspired, no chartjunk)
- OK: muted green #4a8
- Critical: saturated red #e44
- Warning: amber #da2
- Info: slate blue #68a
- Sparklines: data rendered directly, no axes or labels

## Module structure
- `doctor.py` — 7 diagnostic functions, DiagnosticResult, apply_fixes()
- `widgets.py` — DoctorModal added
- `tui.py` — `D` keybinding
- `styles.tcss` — DoctorModal styles

## Git integration
Pre-fix: git_snapshot with "doctor: pre-fix" tag
Post-fix: git_snapshot with "doctor: post-fix N issues" tag
