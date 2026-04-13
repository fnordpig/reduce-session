# Reduction Pipeline Refactor + Prescription System

**Date**: 2026-04-13
**Status**: Approved

## Problem

`reduction.py` has grown to 3622 lines with 66 top-level definitions. It contains text compression primitives, token budgeting, 12+ detection passes, 10+ strategy implementations, position-aware trimming, LLM compression, and the pipeline orchestrator — all in one file. Adding new strategies requires understanding the entire monolith. The pipeline has no formal composition model — strategies are called inline with ad-hoc ordering.

## Design

### Module Decomposition

```
src/reduce_session/
├── reduction.py          # Pipeline orchestrator + ReductionResult + reduce_session()
├── invariants.py         # (exists) atomic writes, is_protected, relink, locks
├── compression.py        # Text compression primitives
├── token_budget.py       # TokenBudget + CHARS_PER_TOKEN + extract_last_usage
├── detection.py          # Cross-message intelligence (detect_* functions)
├── strategies.py         # All strategy implementations
├── trimming.py           # Position-aware field trimming
├── llm_compression.py    # LLM classify/distill pipeline
├── helpers.py            # Content-block helpers, PROFILES, aggressiveness
└── prescription.py       # Strategy registry + prescription definitions
```

### Strategy Protocol

```python
@dataclass
class StrategyResult:
    stats: dict[str, int]
    dropped_uuids: dict[str, str | None]

@dataclass
class PipelineContext:
    aggr_fn: Callable[[float], float]
    profile: str
    agg_limits: dict
    gen_limits: dict
    tool_id_map: dict
    llm_provider: object | None
    progress_callback: object | None
    # Cross-strategy state
    stale_read_ids: set[str]
    duplicate_blocks: dict
    constant_fields: set[str]
    # ... extensible

@dataclass
class StrategyInfo:
    name: str
    description: str
    tier: str  # "gentle", "standard", "aggressive"
    fn: Callable[[list[dict], PipelineContext], StrategyResult]
```

Strategy signature: `fn(kept_objs, ctx) -> StrategyResult`. Mutates in place. Returns stats + dropped_uuids.

### Prescriptions

Cumulative tiers:

```python
PRESCRIPTIONS = {
    "gentle": [
        "compact-summary-collapse",
        "noise-drop",
        "attribution-snapshot-strip",
        "file-history-dedup",
        "metadata-strip",
    ],
    "standard": [
        # gentle strategies, then:
        "stale-reads-detect",
        "duplicate-blocks-detect",
        "error-retry-collapse",
        "constant-envelope-detect",
        "system-reminder-dedup",
        "stale-read-trim",
        "envelope-strip",
        "tool-result-age",
        "edit-sequence-collapse",
        "dead-persisted-output-replace",
    ],
    "aggressive": [
        # standard strategies, then:
        "http-spam-collapse",
        "image-strip",
        "document-dedup",
        "toolUseResult-strip",
        "nuclear-tool-replace",
        "mega-block-trim",
    ],
}
```

### Pipeline Orchestrator

```python
def reduce_session(path, profile="standard", ...):
    parsed = [json.loads(line) for line in lines]
    ctx = PipelineContext(...)
    strategy_names = resolve_prescription(profile)

    for name in strategy_names:
        result = STRATEGIES[name].fn(kept_objs, ctx)
        stats.update(result.stats)
        if result.dropped_uuids:
            relink_parent_chains(kept_objs, result.dropped_uuids)

    _position_aware_trim(kept_objs, ctx)

    if ctx.llm_provider is not None:
        _llm_compression_pass(kept_objs, ctx)

    kept_objs, orphan_count = fix_orphaned_tool_results(kept_objs)
    return ReductionResult(...)
```

### Key Decisions

1. **Strategies call aggr_fn themselves** — position stays fresh after drops.
2. **LLM is a separate phase** after all heuristic strategies. Not a strategy.
3. **Relink after each drop, orphan-fix once at end**.
4. **Position-aware trimming** is not a strategy — it's the core U-curve pass.
5. **Backward-compatible re-exports** — `from reduce_session.reduction import X` keeps working.

### Detection → Action Wiring

Detection strategies populate `ctx` fields. Action strategies consume them. Prescription ordering guarantees detections precede consumers:

- `stale-reads-detect` → `ctx.stale_read_ids` → consumed by `stale-read-trim`
- `duplicate-blocks-detect` → `ctx.duplicate_blocks` → consumed by position-aware trim
- `constant-envelope-detect` → `ctx.constant_fields` → consumed by `envelope-strip`

### New Features in This Refactor

| # | Item | Module |
|---|---|---|
| 1 | toolUseResult full strip | strategies.py (aggressive tier) |
| 3 | Cross-message system-reminder dedup | strategies.py (standard tier) |
| 4 | Microcompact awareness | trimming.py |
| 8 | Protected-type-survival doctor check | doctor.py |
| 9 | Reduction integrity self-test | tests/test_reduction_integrity.py |
| 11 | SKILL.md update | SKILL.md |
