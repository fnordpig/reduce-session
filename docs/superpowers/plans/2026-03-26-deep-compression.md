# Deep Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Squeeze the remaining 80% of context content that current compression misses — tool_results, tool_use inputs, Edit diffs, Agent prompts, and redundant Reads.

**Architecture:** Five proposals implemented in reduction.py: (1) tighten truncation limits indexed to g/s/a, (2) nuclear middle-zone tool content replacement at aggr > 0.8, (3) extend LLM distillation to tool_result content and Agent prompts, (4) collapse consecutive Edit sequences per file, (5) Read result deduplication. All respect the U-curve gradient.

**Tech Stack:** Python, existing reduction.py pipeline, LLM provider abstraction.

---

## File Structure

```
src/reduce_session/
    reduction.py        — All 5 proposals (PROFILES limits, nuclear replace, detect functions, LLM pipeline)
    llm/prompts.py      — New type-specific prompts for tool_result distillation
tests/
    test_deep_compression.py  — Tests for all 5 proposals
```

---

### Task 1: Tighten truncation limits for all three profiles

**Files:**
- Modify: `src/reduce_session/reduction.py:14-145` (PROFILES dict)
- Create: `tests/test_deep_compression.py`

The current aggressive limits are too generous. Tighten across all three profiles:

- [ ] **Step 1: Write tests**

```python
# tests/test_deep_compression.py
import json
import pytest
from pathlib import Path

from reduce_session.reduction import PROFILES, reduce_session


def test_aggressive_limits_are_tight():
    """Aggressive profile limits should be ≤500 for most tool content."""
    agg = PROFILES["aggressive"]["aggressive"]
    assert agg["Bash"] <= 500
    assert agg["Read"] <= 500
    assert agg["Edit"] <= 300
    assert agg["Agent"] <= 500
    assert agg["default"] <= 500
    assert agg["tool_input.Edit"] <= 200
    assert agg["tool_input.Agent"] <= 300


def test_standard_limits_are_moderate():
    agg = PROFILES["standard"]["aggressive"]
    assert agg["Bash"] <= 1000
    assert agg["Read"] <= 1000


def test_gentle_limits_are_generous():
    agg = PROFILES["gentle"]["aggressive"]
    assert agg["Bash"] >= 1500
    assert agg["Read"] >= 2000
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_deep_compression.py -v
```

- [ ] **Step 3: Update PROFILES in reduction.py**

New aggressive-side limits indexed to g/s/a profiles:

**aggressive profile** (lines 14-39):
```python
"aggressive": {
    "aggressive": {
        "Bash": 400, "Read": 400, "Agent": 500, "Write": 400,
        "Edit": 300, "mcp": 800, "default": 400,
        "tur.originalFile": 100, "tur.stdout": 400, "tur.content": 400,
        "tur.oldString": 200, "tur.newString": 200, "tur.file": 400,
        "tool_input.Write": 200, "tool_input.Edit": 200,
        "tool_input.Agent": 300, "thinking": 0, "user_text": 800,
    },
    "gentle": {
        "Bash": 3000, "Read": 4000, "Agent": 6000, "Write": 2000,
        "Edit": 1500, "mcp": 6000, "default": 3000,
        "tur.originalFile": 400, "tur.stdout": 3000, "tur.content": 2000,
        "tur.oldString": 1500, "tur.newString": 1500, "tur.file": 2000,
        "tool_input.Write": 2000, "tool_input.Edit": 1500,
        "tool_input.Agent": 3000, "thinking": 2000, "user_text": 6000,
    },
},
```

**standard profile** (lines 61-101):
```python
"standard": {
    "aggressive": {
        "Bash": 800, "Read": 800, "Agent": 1000, "Write": 600,
        "Edit": 500, "mcp": 2000, "default": 800,
        "tur.originalFile": 200, "tur.stdout": 800, "tur.content": 600,
        "tur.oldString": 500, "tur.newString": 500, "tur.file": 600,
        "tool_input.Write": 600, "tool_input.Edit": 500,
        "tool_input.Agent": 800, "thinking": 0, "user_text": 1500,
    },
    "gentle": {
        "Bash": 4000, "Read": 6000, "Agent": 8000, "Write": 3000,
        "Edit": 2000, "mcp": 10000, "default": 6000,
        "tur.originalFile": 800, "tur.stdout": 4000, "tur.content": 3000,
        "tur.oldString": 2000, "tur.newString": 2000, "tur.file": 3000,
        "tool_input.Write": 3000, "tool_input.Edit": 2000,
        "tool_input.Agent": 4000, "thinking": 3000, "user_text": 8000,
    },
},
```

**gentle profile** (lines 103-144): keep current values (already generous).

- [ ] **Step 4: Run tests — verify they pass**
- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -q
```

- [ ] **Step 6: Verify on whale**

```bash
uv run python -c "
from reduce_session.reduction import reduce_session
r = reduce_session('/Users/rwaugh/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl', profile='aggressive', estimate_tokens=True)
print(f'Saved: {(r.orig_size - r.new_size)/1024/1024:.1f} MB ({(r.orig_size - r.new_size)/r.orig_size*100:.0f}%)')
"
```

- [ ] **Step 7: Commit**

```bash
git commit -am "feat: tighten truncation limits — aggressive Bash/Read/Edit ≤500 chars"
```

---

### Task 2: Nuclear middle-zone tool content replacement

**Files:**
- Modify: `src/reduce_session/reduction.py` — new function `nuclear_tool_replace()`, wire into Pass 3.5
- Modify: `tests/test_deep_compression.py`

At aggr > 0.8 (the plateau of the U-curve, ~20% of the session), replace ALL tool content with one-line summaries. The model has near-zero recall here anyway.

- [ ] **Step 1: Write tests**

```python
def test_nuclear_tool_replace_read(sample_session):
    """Read tool_results in deep middle zone should be replaced with one-liners."""
    r = reduce_session(str(sample_session), profile="aggressive")
    # Check that some tool_results became one-line summaries
    found_replacement = False
    for line in r.kept_lines:
        if "[Read:" in line and "lines]" in line:
            found_replacement = True
            break
    assert found_replacement or r.stats.get("nuclear_reads_replaced", 0) > 0


def test_nuclear_tool_replace_respects_gentle_zone(sample_session):
    """Nuclear replacement should NOT happen in gentle zones."""
    r = reduce_session(str(sample_session), profile="gentle")
    assert r.stats.get("nuclear_reads_replaced", 0) == 0
```

- [ ] **Step 2: Implement `nuclear_tool_replace`**

Add to reduction.py after `detect_superseded_edits`:

```python
def nuclear_tool_replace(kept_objs, aggr_fn, tool_id_map):
    """In the deep middle zone (aggr > 0.8), replace ALL tool content with one-line summaries.

    Returns stats dict with counts of replacements.
    """
    stats = {}
    total = len(kept_objs)
    reads_replaced = 0
    bash_replaced = 0
    edits_replaced = 0

    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        if aggr <= 0.8:
            continue

        t = get_msg_type(obj)
        msg = obj.get("message", {})
        content = msg.get("content")

        if t == "user" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    tool_name = tool_id_map.get(tool_id, "")
                    inner = block.get("content", "")
                    if isinstance(inner, str) and len(inner) > 200:
                        if tool_name in ("Read", "read"):
                            lines_count = inner.count("\n") + 1
                            block["content"] = f"[Read: {lines_count} lines]"
                            reads_replaced += 1
                        elif tool_name in ("Bash", "bash"):
                            # Keep first 100 chars + exit code if present
                            preview = inner[:100].replace("\n", " ")
                            block["content"] = f"[Bash: {preview}...]"
                            bash_replaced += 1

        if t == "assistant" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                inp = block.get("input", {})
                if not isinstance(inp, dict):
                    continue
                if name in ("Edit", "edit"):
                    old = inp.get("old_string", "")
                    new = inp.get("new_string", "")
                    if len(old) + len(new) > 200:
                        fp = inp.get("file_path", "?")
                        inp["old_string"] = f"[~{len(old)} chars]"
                        inp["new_string"] = f"[~{len(new)} chars]"
                        edits_replaced += 1

    if reads_replaced:
        stats["nuclear_reads_replaced"] = reads_replaced
    if bash_replaced:
        stats["nuclear_bash_replaced"] = bash_replaced
    if edits_replaced:
        stats["nuclear_edits_replaced"] = edits_replaced
    return stats
```

- [ ] **Step 3: Wire into reduce_session() after Pass 3.5**

After the semantic elision block and before the LLM pass:

```python
    # -- Pass 3.55: Nuclear tool content replacement (deep middle zone) --
    nuclear_stats = nuclear_tool_replace(kept_objs, aggr_fn, tool_id_map)
    stats.update(nuclear_stats)
```

- [ ] **Step 4: Run tests, verify on whale, commit**

---

### Task 3: Extend LLM distillation to tool_result content and Agent prompts

**Files:**
- Modify: `src/reduce_session/reduction.py` — expand `_llm_compression_pass` distill worker
- Modify: `src/reduce_session/llm/prompts.py` — add tool_result distillation prompts
- Modify: `tests/test_deep_compression.py`

Currently the LLM only distills assistant text (19% of content). Extend to tool_result strings and Agent tool_use prompts (41% more).

- [ ] **Step 1: Add tool_result and Agent distillation prompts to prompts.py**

Add to each profile's `type_prompts` dict:

```python
# For all profiles:
"TOOL_RESULT_BASH": "Summarize this command output to its key result...",
"TOOL_RESULT_READ": "What was learned from reading this file? One sentence...",
"TOOL_RESULT_AGENT": "Summarize what this agent accomplished...",
"TOOL_RESULT_DEFAULT": "Summarize this tool output to its key finding...",
"AGENT_PROMPT": "Summarize the task being dispatched in one sentence...",
```

Gentle versions keep more detail, aggressive versions are ruthless.

- [ ] **Step 2: Expand the distill worker in `_llm_compression_pass`**

After the existing distill_worker processes DISTILL-classified assistant text, add a second phase that processes:
1. tool_result content > 200 chars in DISTILL-classified exchanges
2. Agent tool_use prompt strings > 200 chars

The distill worker queue should also receive tool_result items:

```python
# In classify_worker, after classifying each exchange:
if route == Route.DISTILL:
    # Queue assistant text for distillation
    await distill_queue.put((pos, obj, cat, "assistant_text"))
    # Also queue tool_result content for distillation
    await distill_queue.put((pos, obj, cat, "tool_results"))
```

The distill worker then handles both types:

```python
async def distill_worker():
    ...
    while True:
        item = await distill_queue.get()
        if item is None:
            break
        pos, obj, cat, target = item

        if target == "assistant_text":
            # Existing logic
            text = _extract_assistant_text(obj)
            ...
        elif target == "tool_results":
            # New: distill tool_result blocks
            _distill_tool_results(kept_objs[pos], provider, cat, profile)
```

Add helper `_distill_tool_results` that iterates user message tool_result blocks and distills each one > 200 chars.

For Agent prompts: in the scaffold strip phase, also process assistant tool_use blocks where `name == "Agent"` and `len(inp.get("prompt", "")) > 200`.

- [ ] **Step 3: Tests, verify, commit**

---

### Task 4: Collapse consecutive Edit sequences per file

**Files:**
- Modify: `src/reduce_session/reduction.py` — new `collapse_edit_sequences()` function
- Modify: `tests/test_deep_compression.py`

When a file is edited N times in the middle zone, instead of N individual Edit tool_use blocks with full old/new strings, collapse to one summary: `[N edits to file.rs: summary of changes]`.

This differs from `detect_superseded_edits` which only marks edits as superseded — this collapses the SEQUENCE into one entry.

- [ ] **Step 1: Write tests**

```python
def test_collapse_edit_sequences():
    from reduce_session.reduction import collapse_edit_sequences, make_aggressiveness_fn

    aggr_fn = make_aggressiveness_fn(10, 75)
    objs = []
    # 5 edits to the same file in the middle zone
    for i in range(5):
        objs.append({
            "type": "assistant", "uuid": f"a{i}", "parentUuid": f"u{i}",
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Edit", "id": f"t{i}",
                 "input": {"file_path": "/foo/bar.rs",
                           "old_string": f"old content {i} " * 50,
                           "new_string": f"new content {i} " * 50}}
            ]},
        })
        # interleave with user tool_result
        objs.append({
            "type": "user", "uuid": f"u{i+1}", "parentUuid": f"a{i}",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": "ok"}
            ]},
        })

    stats = collapse_edit_sequences(objs, aggr_fn)
    # First 4 edits should be collapsed, last one kept
    assert stats.get("edit_sequences_collapsed", 0) >= 3
```

- [ ] **Step 2: Implement `collapse_edit_sequences`**

```python
def collapse_edit_sequences(kept_objs, aggr_fn):
    """Collapse consecutive edits to the same file in the middle zone.

    For files with 3+ edits, keep only the last Edit's full content.
    Replace earlier Edits with a one-line summary.
    """
    total = len(kept_objs)
    # Build: file_path -> [(pos, block_index)] for edits in middle zone
    file_edits = {}
    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        if aggr <= 0.3:
            continue
        if get_msg_type(obj) != "assistant":
            continue
        for bi, block in enumerate(get_content_blocks(obj)):
            if block.get("type") == "tool_use" and block.get("name") in ("Edit", "edit"):
                inp = block.get("input", {})
                fp = inp.get("file_path", "")
                if fp:
                    file_edits.setdefault(fp, []).append((pos, bi))

    collapsed = 0
    for fp, edits in file_edits.items():
        if len(edits) < 3:
            continue
        edits.sort(key=lambda x: x[0])
        # Collapse all but the last
        for pos, bi in edits[:-1]:
            block = get_content_blocks(kept_objs[pos])[bi]
            inp = block.get("input", {})
            old_len = len(inp.get("old_string", ""))
            new_len = len(inp.get("new_string", ""))
            if old_len + new_len > 100:
                inp["old_string"] = ""
                inp["new_string"] = f"[collapsed: ~{old_len + new_len} chars, see later edit]"
                collapsed += 1

    return {"edit_sequences_collapsed": collapsed} if collapsed else {}
```

- [ ] **Step 3: Wire into reduce_session() in Pass 3.5 area, tests, commit**

---

### Task 5: Read result deduplication

**Files:**
- Modify: `src/reduce_session/reduction.py` — new `dedup_read_results()` function
- Modify: `tests/test_deep_compression.py`

If the same file was Read multiple times, only the LAST Read needs content. Earlier Reads get replaced with `[Read: path - content superseded by later read]`.

- [ ] **Step 1: Write tests**

```python
def test_dedup_read_results():
    from reduce_session.reduction import dedup_read_results

    objs = []
    # Read the same file 3 times
    for i in range(3):
        objs.append({
            "type": "assistant", "uuid": f"a{i}", "parentUuid": f"u{i}",
            "message": {"role": "assistant", "content": [
                {"type": "tool_use", "name": "Read", "id": f"r{i}",
                 "input": {"file_path": "/foo/bar.rs"}}
            ]},
        })
        objs.append({
            "type": "user", "uuid": f"u{i+1}", "parentUuid": f"a{i}",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"r{i}",
                 "content": f"file content version {i} " * 100}
            ]},
        })

    stats = dedup_read_results(objs)
    assert stats.get("reads_deduped", 0) == 2  # first 2 replaced
    # Last read should still have full content
    last_result = objs[-1]["message"]["content"][0]["content"]
    assert "version 2" in last_result
```

- [ ] **Step 2: Implement `dedup_read_results`**

```python
def dedup_read_results(kept_objs):
    """If the same file was Read multiple times, keep only the last Read's content."""
    # Map: tool_use_id -> (file_path, position)
    read_uses = {}
    for pos, obj in enumerate(kept_objs):
        for block in get_content_blocks(obj):
            if block.get("type") == "tool_use" and block.get("name") in ("Read", "read"):
                inp = block.get("input", {})
                fp = inp.get("file_path", "")
                tid = block.get("id", "")
                if fp and tid:
                    read_uses[tid] = (fp, pos)

    # Group by file_path, find all reads per file
    file_reads = {}  # fp -> [(tool_id, pos)]
    for tid, (fp, pos) in read_uses.items():
        file_reads.setdefault(fp, []).append((tid, pos))

    # For files read multiple times, mark all but last as superseded
    superseded_ids = set()
    for fp, reads in file_reads.items():
        if len(reads) < 2:
            continue
        reads.sort(key=lambda x: x[1])
        for tid, pos in reads[:-1]:
            superseded_ids.add(tid)

    # Replace superseded Read results
    deduped = 0
    for obj in kept_objs:
        if get_msg_type(obj) != "user":
            continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tid = block.get("tool_use_id", "")
            if tid in superseded_ids:
                fp = read_uses.get(tid, ("?", 0))[0]
                inner = block.get("content", "")
                line_count = inner.count("\n") + 1 if isinstance(inner, str) else 0
                block["content"] = f"[Read: {_strip_non_ascii(fp)} - {line_count} lines, superseded by later read]"
                deduped += 1

    return {"reads_deduped": deduped} if deduped else {}
```

- [ ] **Step 3: Wire into reduce_session() in Pass 3 area (before semantic elision), tests, commit**

---

### Task 6: Integration testing and push

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v
```

- [ ] **Step 2: Test on pristine whale**

```bash
uv run python -c "
from reduce_session.reduction import reduce_session
BAK = '/Users/rwaugh/.claude/projects/-Users-rwaugh-src-mine-ripvec/db776eab-e7c2-4e9d-8855-28294c27b5db.jsonl.bak2'
for profile in ['gentle', 'standard', 'aggressive']:
    r = reduce_session(BAK, profile=profile, estimate_tokens=True)
    saved_mb = (r.orig_size - r.new_size)/1024/1024
    pct = (r.orig_size - r.new_size)/r.orig_size*100
    print(f'{profile:>12s}: {r.orig_size/1024/1024:.1f} -> {r.new_size/1024/1024:.1f} MB ({pct:.0f}% saved)')
    for k in ['nuclear_reads_replaced', 'nuclear_bash_replaced', 'nuclear_edits_replaced',
              'edit_sequences_collapsed', 'reads_deduped']:
        v = r.stats.get(k, 0)
        if v:
            print(f'              {k}: {v}')
"
```

- [ ] **Step 3: Test on current whale (already reduced)**

Same command but with the current file. Should now show meaningful savings even on already-reduced content.

- [ ] **Step 4: Commit and push**

```bash
git add -A
git commit -m "feat: deep compression — 5 proposals for squeezing remaining 80% of context"
git push origin main
```
