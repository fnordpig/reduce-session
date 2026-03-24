---
name: reduce-session
description: >
  Diagnose and reduce bloated Claude Code session JSONL files to recover context space.
  Use this skill when the user mentions context being full, wants to slim down a session,
  asks about reducing JSONL files, says "/context shows too much usage", mentions stale
  token counts, or wants to resume a session without auto-compaction destroying history.
  Also trigger when the user mentions context percentage being wrong, session files being
  too large, or wanting to preserve conversation quality while freeing space.
---

# Reduce Session Context

Claude Code session history lives in append-only JSONL files under `~/.claude/projects/`. Over long sessions these grow to 20MB+, tokenizing to millions of tokens — far beyond the context window. Claude Code auto-compacts on resume, but compaction is lossy and opaque. This skill lets you surgically reduce session files while preserving the high-value conversation content.

## When to use this

- `/context` shows high usage (especially if the category breakdown doesn't match the total — that means stale token counters)
- A session file is too large to resume without auto-compaction
- The user wants to continue a long session without losing history

## The Technique

There are three independent problems. All three must be addressed or the reduction won't take effect.

### Problem 1: Bulk metadata (typically 40-60% of file size)

Session JSONL files contain many line types. Most are operational metadata with zero conversation value:

| Type | What it is | Value for context recovery |
|------|-----------|---------------------------|
| `progress` | Agent/hook status pings ("agent running...") | None |
| `file-history-snapshot` | File backup metadata for undo | None |
| `queue-operation` | Internal message queue bookkeeping | None |
| `last-prompt` | Prompt boundary markers | None |
| `system` | System prompt + skill listings (repeated per API call) | Keep one, drop duplicates |
| `user` | Human prompts + tool results | High — but contains noise too |
| `assistant` | Reasoning + tool calls | Highest — never cut |

**Action**: Drop `progress`, `file-history-snapshot`, `queue-operation`, `last-prompt` entirely. Deduplicate `system` messages by content hash.

Within `user` messages, drop:
- **Task notifications** (`<task-notification>`) — delivery receipts for background tasks
- **Local command noise** (`/reload-plugins`, `/plugin`, `/mcp`, `/login`, `/effort`, `/compact`) — UI commands with no conversational content

### Problem 2: Bloated fields (typically 20-30% of file size)

Several fields store redundant or oversized data:

**`toolUseResult`** — A top-level field on `user` lines that Claude Code uses for its own bookkeeping. Contains:
- `originalFile`: The **entire file** before each Edit. A 50KB file edited 16 times = 800KB of redundant snapshots. Truncate aggressively.
- `stdout`: Bash output duplicated from `message.content`. Clean and truncate.
- `content`: Full file content from Write operations. Truncate.
- `structuredPatch`: A diff representation. Keep (it's an array — deleting it crashes the parser) but trim long patch lines.

**Tool result content** — Truncate long Bash output, file reads, and MCP results to head+tail (3-4KB). The model can re-read files from disk if needed.

**Tool call inputs** — Write `content`, Edit `old_string`/`new_string`, Agent `prompt` can all be truncated. The source of truth is the filesystem.

**Shell noise** — ASCII art banners from shell init (`____`, `/ /`, `/_/` patterns) appear in every Bash result. Strip them. Also collapse cargo compile lines (`Compiling`, `Downloading`, `Fresh`) to summaries.

### Problem 3: Stale token counter (the invisible blocker)

This is the non-obvious one. Each assistant message includes `message.usage` from the API response:

```json
{"message": {"usage": {"cache_read_input_tokens": 793690, ...}}}
```

`/context` reads the **last** `message.usage` as the current token count. After reduction, this stale value persists — `/context` will show the old count even though the actual content is much smaller.

**Action**: Delete `message.usage` from all assistant messages **except the last main-chain assistant message**. Claude Code needs the final usage to bootstrap token tracking on resume. Stripping all of them causes it to lose the conversation.

### Problem 4: Broken parentUuid chain (the session killer)

Claude Code reconstructs conversations as a tree via `parentUuid` pointers. Every message has a `uuid` and a `parentUuid` linking to the previous message. When you drop a line, any message that pointed to it as a parent becomes an orphan — Claude Code can't find it during tree reconstruction and the conversation appears empty except for the leaf.

**Action**: Before dropping any line, record its `uuid → parentUuid` mapping. After dropping, walk every kept message's `parentUuid` — if it points to a dropped UUID, follow the chain up through dropped nodes until you find a surviving ancestor. This splices dropped nodes out of the tree cleanly.

```python
# When dropping: record uuid → parentUuid
dropped[obj["uuid"]] = obj["parentUuid"]

# After dropping: reparent kept messages
for obj in kept:
    parent = obj.get("parentUuid")
    while parent in dropped:
        parent = dropped[parent]
    obj["parentUuid"] = parent
```

This is the difference between "session loads correctly" and "session shows only the last message."

### Problem 5: Multiple overlapping files

Claude Code creates continuation files during compaction: `session_id.TIMESTAMP.jsonl`. These can overlap with the main file. Check:

```bash
ls session_id*.jsonl | grep -v .bak
```

If multiple files exist, they may share UUIDs. Reduce all active files, not just the main one.

## Safety Rules

These are hard-won from breaking Claude Code's parser and session loading:

1. **Never delete keys from content blocks** — set string values to truncated versions instead. The parser expects specific keys to exist.
2. **Never change types** — if a field is an array, keep it as an array. If it's a string, keep it as a string. `structuredPatch` deletion caused `undefined is not an object (evaluating 'K.reduce')`.
3. **Always reparent when dropping lines** — every dropped message's children must be reparented to the dropped message's parent, or the `parentUuid` tree breaks and Claude Code only sees the last message.
4. **Strip all `message.usage` fields** — these contain stale pre-reduction token counts that `/context` displays as current. The "only sees last message" bug on resume is caused by broken `parentUuid` chains (rule 3), not missing usage. Strip usage from all assistant messages.
5. **Never strip `signature` from thinking blocks** — the API requires `signature` on any thinking block that's kept. Stripping it causes `400: thinking.signature: Field required`. Only remove signatures by removing the entire thinking block.
6. **Preserve `tool_reference` blocks** — `tool_result.content` lists can contain both `text` and `tool_reference` items. Only modify `text` items.
6. **Deep-copy before mutating** — `json.loads(json.dumps(obj))` or equivalent.
7. **Back up before replacing** — always `cp file.jsonl file.jsonl.bak` first.

## Running the Reduction

The bundled script handles all of the above:

```bash
# Back up first
cp session.jsonl session.jsonl.bak

# Reduce
python3 ${CLAUDE_SKILL_DIR}/scripts/reduce_session.py session.jsonl

# Replace
mv session.jsonl.reduced session.jsonl
```

The script:
- Takes a JSONL path as argument
- Writes `<path>.reduced` (never overwrites in-place)
- Prints a summary of what was removed and how much was saved
- Preserves all structural invariants

Run it on every active `.jsonl` file for the session (check for continuation files).

## Diagnostic Workflow

If the user just says "context is full" or "reduce this session", follow this sequence:

1. **Find the session files**: `ls ~/.claude/projects/*/SESSION_ID*.jsonl`
2. **Census**: Count lines and bytes by type to understand the weight distribution
3. **Check for stale usage**: Look at the last assistant message's `message.usage.cache_read_input_tokens` — if it's huge, that's the `/context` display source
4. **Check for multiple files**: Look for `SESSION_ID.TIMESTAMP.jsonl` continuation files
5. **Back up all active files**
6. **Run the reduction script** on each active file
7. **Replace** the originals with reduced versions
8. **Have the user resume** the session and check `/context`

## Expected Results

From our reference session (ripvec, 4-day Rust development):

| Metric | Before | After | Retention |
|--------|--------|-------|-----------|
| File size | 23MB | 8.8MB | — |
| `/context` display | 794k (stale) | 32k (real) | — |
| Assistant reasoning | 318KB | 318KB | **100%** |
| Tool calls | 1,895 | 1,895 | **100%** |
| Tool results | 1,885 | 1,885 | **100%** |
| Substantive user prompts | 204 | 204 | **100%** |
| Tool result content | 2,196KB | 1,645KB | 75% |

The 75% tool result retention is from head+tail truncation of long outputs. The model can re-read files from disk if it needs the middle.
