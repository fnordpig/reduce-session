# reduce-session

Surgically reduce Claude Code session JSONL files while preserving conversation quality. Interactive TUI, CLI, and MCP server.

## What It Does

Claude Code sessions live in append-only JSONL files under `~/.claude/projects/`. Long sessions grow to 20MB+, most of it noise: progress ticks, file-history snapshots, stale tool results, duplicate CLAUDE.md injections, and metadata bloat. Claude Code's built-in compaction is lossy and opaque.

reduce-session removes the noise with **18 composable strategies** across 3 prescription tiers, a U-curve aggressiveness gradient matching the "Lost in the Middle" LLM attention pattern, and optional LLM-powered semantic distillation. Your actual conversation, decisions, and working context stay untouched.

### Key Features

- **18 reduction strategies** across gentle (5), standard (10), and aggressive (18) tiers
- **Interactive TUI** — session tree, conversation preview, sparkline density profiles, doctor modal
- **MCP server** — `list_sessions`, `browse_session`, `doctor`, `reduce` callable by Claude
- **LLM semantic distillation** — classify exchanges as keep/distill/strip using any provider
- **14 doctor checks** with auto-fix — diagnose and repair session corruption
- **Compact-summary exploitation** — 85-95% savings on compacted sessions
- **Safety invariants** — atomic writes, fsync, prune locks, conflict detection, protected-message taxonomy
- **Git-backed history** — timestamped backups, reduction timeline, restore from any point

## Install

```bash
# From source
git clone https://github.com/fnordpig/reduce-session.git
cd reduce-session
uv sync

# With LLM providers (pick what you use)
uv sync --extra anthropic    # Claude API
uv sync --extra openai       # OpenAI API
uv sync --extra google       # Gemini API
uv sync --extra local-macos  # MLX local models (Apple Silicon)
uv sync --extra local        # llama.cpp local models
```

## Quick Start

```bash
# Launch the interactive TUI browser
reduce-session --browse

# Analyze a specific session (dry-run by default)
reduce-session ~/.claude/projects/-Users-me-myproject/SESSION_ID.jsonl

# Apply standard reduction with backup
reduce-session SESSION.jsonl --apply

# Aggressive reduction
reduce-session SESSION.jsonl --profile aggressive --apply

# Diagnose session corruption
reduce-session SESSION.jsonl --doctor

# Auto-fix all diagnosed issues
reduce-session SESSION.jsonl --doctor --fix

# Restore from backup
reduce-session SESSION.jsonl --restore

# View reduction history
reduce-session SESSION.jsonl --history

# Token budget breakdown
reduce-session SESSION.jsonl --tokens
```

## Strategies

18 composable strategies organized into cumulative prescription tiers:

### Gentle (5 strategies)

| Strategy | What It Does | Savings |
|---|---|---|
| `compact-summary-collapse` | Drop pre-compaction messages already in the summary | 85-95% |
| `noise-drop` | Drop progress ticks, queue-operations, task notifications | 40-48% |
| `attribution-snapshot-strip` | Remove cost-attribution metadata entries | 0-2% |
| `file-history-dedup` | Keep latest file-history snapshot per messageId | 3-6% |
| `metadata-strip` | Strip token usage, stop_reason, constant envelope fields | 1-3% |

### Standard (adds 5 more)

| Strategy | What It Does | Savings |
|---|---|---|
| `stale-read-detection` | Identify file reads superseded by later edits | 0.5-2% |
| `duplicate-block-detection` | Content-hash dedup of repeated tool results | 1-5% |
| `error-retry-collapse` | Collapse repeated error-retry tool sequences | 0-5% |
| `envelope-strip` | Strip constant fields (cwd, version, gitBranch) from all but first message | 2-4% |
| `tool-result-age` | Age-based compaction: recent=keep, mid=minify JSON, old=stub | 10-40% |

### Aggressive (adds 8 more)

| Strategy | What It Does | Savings |
|---|---|---|
| `image-strip` | Strip old base64 images, keep newest 20% | 1-40% |
| `document-dedup` | Block-level md5 dedup (catches CLAUDE.md re-injection) | 0-44% |
| `http-spam-collapse` | Remove progress ticks from WebFetch/WebSearch runs | 0-2% |
| `edit-sequence-collapse` | Collapse consecutive edits to the same file | 0-5% |
| `dead-output-replace` | Replace references to deleted tool-result files with stubs | 0-2% |
| `nuclear-tool-replace` | Replace all tool content with one-line summaries | 5-20% |
| `mega-block-trim` | Safety net: truncate any content block over 32KB | varies |
| `LLM semantic distillation` | Classify and summarize exchanges using an LLM provider | 10-30% |

### The U-Curve

Reduction aggressiveness follows a U-curve matching LLM attention patterns:

```
Aggressiveness
  1.0 |          ████████████
      |        ██            ██
      |      ██                ██
      |    ██                    ██
  0.0 |████                        ████
      +---+----+----+----+----+----+---→
      0%  cut              fade    100%
          (gentle)  (aggressive)  (gentle)
```

The start and end of conversations get gentle treatment (high recall zones). The middle gets aggressive compression. Controlled by `--cut` and `--fade` parameters.

## LLM Semantic Distillation

The optional LLM pass classifies each exchange as keep/distill/strip, then summarizes the distill candidates:

```bash
# Use Claude API
reduce-session SESSION.jsonl --llm anthropic:claude-sonnet-4-20250514

# Use a local MLX model (Apple Silicon)
reduce-session SESSION.jsonl --llm local:mlx-community/Qwen2.5-Coder-7B-Instruct-4bit

# Use Ollama
reduce-session SESSION.jsonl --llm ollama:qwen2.5-coder:7b

# Use OpenAI
reduce-session SESSION.jsonl --llm openai:gpt-4o-mini
```

LLM distillation runs after all heuristic strategies and respects the U-curve — only mid-zone exchanges are candidates.

## Doctor

14 diagnostic checks with auto-fix:

| Check | Detects | Auto-Fix |
|---|---|---|
| `compaction_summaries` | Orphaned compact summaries (parentUuid=null) | Graft into chain |
| `parent_chain` | Broken parentUuid references | Reparent to nearest predecessor |
| `stale_tokens` | Stale message.usage counters inflating /context | Recalibrate |
| `overlapping_files` | Continuation files sharing UUIDs | Rename to .bak2 |
| `unreduced_metadata` | Metadata bloat not yet reduced | Strip |
| `reduce_tags` | Missing or stale reduction tags | Update |
| `bloated_tur` | Oversized toolUseResult fields (>10KB) | Truncate |
| `orphaned_tool_results` | Dead persisted-output refs + orphaned files on disk | Replace refs + delete files |
| `corrupted_tool_use` | tool_use.name >200 chars (crash corruption) | Parse and repair |
| `corrupted_content_blocks` | Empty tool_use id / tool_result tool_use_id | Drop blocks |
| `cycle_in_parent_chain` | Circular parentUuid references | Sever cycle |
| `null_parentUuid_at_non_root` | Non-root messages with null parentUuid | Reparent |
| `stale_backups` | Accumulated .bak files wasting disk | Delete |
| `protected_type_survival` | Protected messages lost during reduction | Restore from backup |

```bash
# Diagnose
reduce-session SESSION.jsonl --doctor

# Auto-fix (applies fixes in dependency order)
reduce-session SESSION.jsonl --doctor --fix

# Exit codes: 0=ok, 1=warnings, 2=critical, 3=parse failure
```

## MCP Server

Expose reduce-session to Claude as MCP tools:

```bash
reduce-session-mcp
```

Tools: `list_sessions`, `browse_session`, `get_exchange`, `classify_exchange`, `delete_exchange`, `doctor`, `doctor_fix`, `reduce`.

Add to your Claude Code MCP config or use as a plugin server.

## Interactive TUI

```bash
reduce-session --browse
```

Two-pane interface: session tree on the left (grouped by project, with token counts, age colors, and health indicators), conversation preview on the right. Key bindings:

| Key | Action |
|---|---|
| `r` | Open reduce modal for selected session |
| `e` | Browse conversation exchanges |
| `D` | Run doctor diagnostics |
| `h` | View reduction history |
| `^L` | Refresh session list |
| `q` / `Esc` | Quit |

## Safety

- **Atomic writes** — write to `.tmp`, `fsync`, then `os.replace`. No partial writes.
- **Prune locks** — `fcntl.LOCK_EX` prevents concurrent TUI/CLI/MCP from racing on the same file.
- **Conflict detection** — `FileSnapshot` detects if Claude Code appended lines during reduction. Merges appended content or aborts on conflict.
- **Protected message taxonomy** — compact summaries, boundaries, marble-origami state, task-summary, worktree-state, and isVisibleInTranscriptOnly entries are never modified.
- **Dual-chain reparenting** — both `parentUuid` and `logicalParentUuid` are relinked through dropped messages with cycle detection. Chain exhaustion preserves the original value (never writes null).
- **Timestamped backups** — `.bak` created before every apply. Restore from backup or git tag.
- **Git-backed history** — `--init` creates a git repo in the session directory for full reduction timeline.

## Architecture

```
src/reduce_session/
├── cli.py              # CLI entry point
├── reduction.py        # Pipeline orchestrator + ReductionResult
├── strategies.py       # 18 strategy implementations
├── detection.py        # Cross-message intelligence (detect_* passes)
├── compression.py      # Text compression (structural, stochastic, entropy)
├── trimming.py         # Position-aware field trimming (U-curve)
├── helpers.py          # PROFILES, content-block helpers, aggressiveness
├── token_budget.py     # Token estimation + budget tracking
├── invariants.py       # Atomic writes, is_protected, relink, locks
├── llm_compression.py  # LLM classify/distill pipeline
├── doctor.py           # 14 diagnostic checks + auto-fix
├── session.py          # Session discovery + metadata extraction
├── git_ops.py          # Apply/restore/history via git + .bak
├── mcp_server.py       # FastMCP server
├── tui.py              # Textual app
├── widgets.py          # TUI widgets (browser, doctor modal, reduce modal)
└── llm/                # LLM provider adapters
    ├── anthropic_provider.py
    ├── openai_provider.py
    ├── gemini_provider.py
    ├── ollama.py
    ├── local.py          # MLX + llama.cpp
    ├── prompts.py        # Classification/distillation prompts
    └── base.py           # Provider protocol
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_reduction.py -v

# Run with coverage
uv run pytest tests/ --cov=reduce_session
```

413 tests covering reduction pipeline, strategies, detection passes, doctor checks, invariants, TUI, browser, LLM pipeline, git ops, and session discovery.

## License

MIT
