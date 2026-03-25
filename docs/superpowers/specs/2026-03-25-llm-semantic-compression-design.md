# LLM-Assisted Semantic Compression

## Overview

A two-stage local-first LLM pipeline for reduce-session that strips scaffolding language and summarizes low-value exchanges using small, fast models. Qwen3-1.7B classifies exchanges into routing categories, Qwen3-4B distills DISTILL-routed exchanges and strips scaffolding from all assistant text in the middle zone.

The pipeline slots between existing heuristic elision (Pass 3.5) and structural compression (Pass 4), so structural techniques operate on already-distilled text for maximum value.

## Design Principles

1. **Local-first, batteries included**: `--llm local` auto-downloads Qwen3 GGUF models via `huggingface_hub` and runs inference with MLX (macOS) or llama-cpp-python (Linux). No external processes required.
2. **Multi-provider**: supports local, Ollama, Anthropic, OpenAI, Gemini through a common interface.
3. **Env vars only**: API keys and defaults via environment variables, no config files.
4. **Graceful degradation**: if LLM unavailable, falls back to heuristic-only mode.
5. **Async pipeline**: classification and distillation overlap — 4B starts as soon as first classification batch returns.

## Provider Abstraction

### CLI Interface

```
--llm local                     # MLX (macOS) / llama-cpp (Linux)
--llm ollama:qwen3:4b           # Ollama server
--llm anthropic:haiku           # Claude Haiku
--llm openai:gpt-4o-mini        # OpenAI
--llm gemini:flash              # Gemini
```

No `--llm` flag: check `REDUCE_SESSION_LLM` env var. If unset, no LLM pass (heuristic only).

### Environment Variables

```bash
# Provider selection
REDUCE_SESSION_LLM=local

# Model overrides (local/ollama only)
REDUCE_SESSION_CLASSIFIER=Qwen/Qwen3-1.7B-GGUF
REDUCE_SESSION_DISTILLER=Qwen/Qwen3-4B-GGUF

# API keys (provider-specific, standard env vars)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
OLLAMA_HOST=http://localhost:11434  # default
```

CLI flags override env vars override defaults. No config files for security.

### Provider Protocol

```python
class LLMProvider(Protocol):
    async def classify(self, exchanges: list[dict]) -> list[str]:
        """Classify a batch of exchanges into routing categories.

        Input: list of {"role": str, "text": str, "tool_name": str|None}
        Output: list of category strings, same length
        """
        ...

    async def distill(self, text: str, mode: str) -> str:
        """Distill text content.

        mode="summarize": compress exchange to essential information
        mode="strip_scaffold": remove filler language, keep grounded content
        """
        ...

    async def shutdown(self):
        """Release resources (unload models, close connections)."""
        ...
```

### Provider Implementations

| Provider | Classifier | Distiller | Notes |
|----------|-----------|-----------|-------|
| `local` (macOS) | Qwen3-1.7B via mlx-lm | Qwen3-4B via mlx-lm | Both loaded simultaneously (~3.5GB) |
| `local` (Linux) | Qwen3-1.7B via llama-cpp-python | Qwen3-4B via llama-cpp-python | Both loaded simultaneously |
| `ollama:model` | Same model for both | Same model for both | Single model, mode-switched via prompt |
| `anthropic:haiku` | Haiku | Haiku | Batched API calls |
| `openai:gpt-4o-mini` | gpt-4o-mini | gpt-4o-mini | Batched API calls |
| `gemini:flash` | Gemini Flash | Gemini Flash | Batched API calls |

For API providers, a single model handles both classification and distillation (the model is capable enough). For local, we use the smaller model for classification to save time/memory.

## Model Management

### Local Mode

```python
from huggingface_hub import hf_hub_download

# Standard HF cache: ~/.cache/huggingface/hub/
# Shared with other tools, respects HF_HOME env var
classifier_path = hf_hub_download(
    repo_id="Qwen/Qwen3-1.7B-GGUF",
    filename="Qwen3-1.7B-Q8_0.gguf",
)
distiller_path = hf_hub_download(
    repo_id="Qwen/Qwen3-4B-GGUF",
    filename="Qwen3-4B-Q4_K_M.gguf",
)
```

First run: auto-download with progress bar (~1.8GB + ~2.5GB). Cached after that. If offline or download fails, fall back to heuristic-only with warning.

### Inference Backends

**macOS (MLX)**:
- `mlx-lm` for Metal-accelerated inference
- Both models loaded simultaneously in unified memory (~4.3GB)
- Qwen3-1.7B Q8_0: ~1.8GB, ~80 tok/s on M-series
- Qwen3-4B Q4_K_M: ~2.5GB, ~50 tok/s on M-series

**Linux (llama-cpp-python)**:
- Python bindings for llama.cpp
- CUDA acceleration automatic if available
- CPU fallback with AVX2/NEON

### Model Override

Users can override models via environment variables:

```bash
# Use different quantization
REDUCE_SESSION_CLASSIFIER=Qwen/Qwen3-1.7B-GGUF:qwen3-1.7b-q8_0.gguf
REDUCE_SESSION_DISTILLER=Qwen/Qwen3-4B-GGUF:qwen3-4b-q8_0.gguf

# Use completely different models
REDUCE_SESSION_CLASSIFIER=microsoft/Phi-3-mini-4k-instruct-gguf:phi-3-mini.gguf
```

Format: `repo_id:filename` or just `repo_id` (uses default filename).

## Classification Pipeline (Pass 3.6)

### Categories

| Category | Route | Description |
|----------|-------|-------------|
| DECISION | KEEP | User chose a direction or approach |
| PREFERENCE | KEEP | User stated how they want things done |
| CORRECTION | KEEP | User corrected the model's approach |
| FINDING | KEEP | Diagnostic insight, constraint discovered |
| REASONING | DISTILL | Assistant analysis, explanation, approach discussion |
| IMPLEMENTATION | DISTILL | Code changes, edit sequences, tool operations |
| DIAGNOSTIC | DISTILL | Debugging chain, error investigation |
| AGENT_TRANSCRIPT | DISTILL | Subagent full work log |
| EXPLORATION | HEURISTIC | Reading files, searching, setup |
| SCAFFOLDING | HEURISTIC | Build output, test results, boilerplate |
| ROUTINE | HEURISTIC | Confirmations, status updates, acknowledgments |

### Routing Rules

- **KEEP** (DECISION, PREFERENCE, CORRECTION, FINDING): never touch content. Scaffolding strip still applies to assistant text within these exchanges.
- **DISTILL** (REASONING, IMPLEMENTATION, DIAGNOSTIC, AGENT_TRANSCRIPT): send to 4B for exchange summarization, then scaffold strip.
- **HEURISTIC** (EXPLORATION, SCAFFOLDING, ROUTINE): handled by existing Phase 1 pipeline.

### Batched Classification

Batch 20 exchanges per prompt. The 1.7B responds with a JSON array.

```
System: You classify coding session exchanges for compression routing.
Respond with ONLY a JSON array of category strings, one per exchange.

Categories:
DECISION — user chose a direction
PREFERENCE — user stated how they want things
CORRECTION — user corrected the model
FINDING — diagnostic insight, constraint
REASONING — assistant explains/analyzes
IMPLEMENTATION — code changes, edits
DIAGNOSTIC — debugging, error investigation
AGENT_TRANSCRIPT — subagent work log
EXPLORATION — reading files, searching
SCAFFOLDING — build output, test results
ROUTINE — confirmations, status

---
Exchange 1:
[user] Yeah let's use Metal for the backend
---
Exchange 2:
[assistant] Let me start by reading the configuration file to understand
the current Metal setup. I'll analyze the shader compilation pipeline...
---
...

Output: ["DECISION", "REASONING", ...]
```

### Scope

Classification only runs on exchanges in the middle zone (where U-curve aggressiveness > 0.2). Edge zones (start/end) are not classified — they're preserved by the U-curve gradient.

## Distillation Pipeline (Pass 3.7)

### Mode 1: Exchange Summarization

Applied to DISTILL-routed exchanges. The 4B compresses multi-step exchanges to their essence.

```
System: Compress this coding session exchange to essential information only.
Keep: decisions made, facts discovered, constraints, what changed, errors.
Remove: preamble, transitions, restating context, verbose explanation.
Respond with ONLY the compressed text — one concise paragraph, no commentary.

---
[assistant] Let me analyze this carefully. I'll start by examining the Metal
backend code. Looking at the shader compilation pipeline, I can see that the
issue is with FP64 operations — Metal doesn't support them natively. We'll
need to restructure the BERT attention computation to use FP16 throughout.
Let me also check if the embedding layer needs similar changes...
---

→ Metal doesn't support FP64. Restructuring BERT attention to FP16.
  Embedding layer also needs FP16 conversion.
```

### Mode 2: Scaffolding Strip

Applied to ALL assistant text blocks in the middle zone, including KEEP exchanges. Strips filler language while preserving every grounded fact.

```
System: Strip filler language from this text. Keep ONLY grounded, factual content.
Remove: "Let me", "I'll now", "Looking at this", "Based on what I see",
all transitions, preamble, hedging, restating what user said.
Preserve: every fact, decision, error message, code reference, constraint, number.
Return ONLY the stripped text, nothing else.

---
Let me check the configuration. I see that the batch size is set to 64,
which is causing OOM on the M2's 16GB unified memory. I'll reduce it to
32 which should fit within the available memory budget.
---

→ Batch size 64 causes OOM on M2 16GB. Reducing to 32.
```

### Processing Order

1. Exchange summarization (Mode 1) runs first on DISTILL exchanges
2. Scaffolding strip (Mode 2) runs on ALL assistant text in middle zone
3. This order ensures Mode 2 operates on already-summarized text for DISTILL exchanges

## Async Pipeline Architecture

```python
async def llm_compression_pass(kept_objs, aggr_fn, provider):
    """Pass 3.6 + 3.7: Classification then distillation with overlap."""

    total = len(kept_objs)

    # Identify middle-zone exchanges
    middle = []
    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        if aggr > 0.2:  # in the compressible zone
            middle.append((pos, obj, aggr))

    if not middle:
        return

    # Phase 1: Batched classification with async distillation overlap
    distill_queue = asyncio.Queue()

    async def classify_worker():
        for batch in batched(middle, 20):
            exchange_texts = [extract_exchange_text(obj) for _, obj, _ in batch]
            categories = await provider.classify(exchange_texts)
            for (pos, obj, aggr), cat in zip(batch, categories):
                if cat in {"REASONING", "IMPLEMENTATION", "DIAGNOSTIC", "AGENT_TRANSCRIPT"}:
                    await distill_queue.put((pos, obj, aggr, "summarize"))
        await distill_queue.put(None)  # sentinel

    async def distill_worker():
        while True:
            item = await distill_queue.get()
            if item is None:
                break
            pos, obj, aggr, mode = item
            text = extract_assistant_text(obj)
            if text:
                summary = await provider.distill(text, mode="summarize")
                replace_assistant_text(kept_objs[pos], summary)

    await asyncio.gather(classify_worker(), distill_worker())

    # Phase 2: Scaffolding strip on ALL assistant text in middle zone
    # (sequential — operates on already-summarized content)
    for pos, obj, aggr in middle:
        text = extract_assistant_text(obj)
        if text and len(text) > 50:  # skip tiny blocks
            stripped = await provider.distill(text, mode="strip_scaffold")
            replace_assistant_text(kept_objs[pos], stripped)
```

### Performance Characteristics

| Operation | Local (M-series) | API (Haiku) |
|-----------|-------------------|-------------|
| Classify 200 exchanges (10 batches) | ~30s | ~5s |
| Distill 100 exchanges | ~60s | ~10s |
| Scaffold strip 300 blocks | ~45s | ~8s |
| **Total wall time** | **~2 min** | **~20s** |

Local is slower but free and private. The async overlap reduces wall time by ~30% vs sequential.

## Pipeline Integration

### Insertion Point

```
Pass 3.5:  Heuristic elision (existing, free, deterministic)
Pass 3.6:  LLM classification (1.7B, batched)              ← NEW
Pass 3.7:  LLM distillation + scaffold strip (4B, async)   ← NEW
Pass 4:    Structural compression (existing)
           Now operates on distilled text → char drops and
           minification get more value per byte
Pass 5:    Orphan repair (existing)
```

### Conditional Execution

Pass 3.6/3.7 only run when `--llm` is specified (or `REDUCE_SESSION_LLM` is set). Without LLM, the pipeline is unchanged — heuristic elision + structural compression only.

### Stats Tracking

New stats keys added to ReductionResult:
```python
stats["llm_classified"] = 847
stats["llm_classified_keep"] = 312
stats["llm_classified_distill"] = 289
stats["llm_classified_heuristic"] = 246
stats["llm_distilled"] = 289
stats["llm_scaffold_stripped"] = 601
stats["llm_chars_saved"] = 347291
stats["llm_provider"] = "local:mlx"
stats["llm_classifier_model"] = "Qwen3-1.7B-Q8_0"
stats["llm_distiller_model"] = "Qwen3-4B-Q4_K_M"
```

## Module Structure

```
src/reduce_session/
    llm/
        __init__.py          # create_provider(), LLM_AVAILABLE flag
        base.py              # LLMProvider protocol, categories, routing map
        prompts.py           # Classification and distillation prompt templates
        local.py             # MLX + llama-cpp-python, hf_hub_download
        ollama.py            # Ollama HTTP client
        anthropic_provider.py  # Anthropic API
        openai_provider.py   # OpenAI API
        gemini_provider.py   # Gemini API
    reduction.py             # Pass 3.6/3.7 integration
    cli.py                   # --llm flag, env var handling
    widgets.py               # LLM compression section in modal
```

### Dependency Management

```toml
[project]
dependencies = ["textual>=1.0", "huggingface-hub>=0.20"]

[project.optional-dependencies]
local-macos = ["mlx-lm>=0.10"]
local = ["llama-cpp-python>=0.2"]
anthropic = ["anthropic>=0.40"]
openai = ["openai>=1.0"]
google = ["google-genai>=1.0"]
all-local = ["mlx-lm>=0.10", "llama-cpp-python>=0.2"]
all-api = ["anthropic>=0.40", "openai>=1.0", "google-genai>=1.0"]
```

Installation:
```bash
uv pip install reduce-session[local-macos]    # Mac batteries-included
uv pip install reduce-session[local]          # Linux batteries-included
uv pip install reduce-session[anthropic]      # Haiku via API
uv pip install reduce-session[all-api]        # All API providers
```

## TUI Integration

### Reduce Modal

New "LLM Compression" section in the strategies display:

```
── LLM Compression (Qwen3-1.7B → Qwen3-4B, local:mlx) ──
  classified             847 exchanges
    → KEEP               312 (37%)
    → DISTILL            289 (34%)
    → HEURISTIC          246 (29%)
  distilled              289 exchanges
  scaffold stripped      601 text blocks
  chars saved          347,291
```

### Progress Feedback

During LLM passes, the modal spinner shows:
```
Classifying exchanges... (batch 4/10)
Distilling... (47/289 exchanges)
Stripping scaffolding... (201/601 blocks)
```

## Safety Guarantees

1. **LLM is optional**: no `--llm` flag = no LLM pass. Existing pipeline unchanged.
2. **Heuristics run first**: Phase 1 elision handles the obvious cases before LLM sees anything.
3. **U-curve respected**: LLM only operates in the middle zone. Edge zones preserved.
4. **Git-backed**: reductions are committed with tags, restorable.
5. **Deterministic fallback**: if LLM produces garbage (empty, error, timeout), keep the original text.
6. **No data leaves machine in local mode**: MLX/llama-cpp run entirely locally.
7. **Scaffolding strip preserves facts**: the prompt explicitly instructs "keep every fact, decision, error message, code reference, constraint, number."

## Error Handling

| Failure | Action |
|---------|--------|
| Model download fails | Warning, fall back to heuristic-only |
| MLX not available on macOS | Try llama-cpp-python, then heuristic |
| Classification returns wrong count | Retry batch once, then classify as KEEP (safe default) |
| Distillation returns empty | Keep original text |
| Distillation returns longer than input | Keep original text |
| API rate limit | Backoff + retry, then fall back to heuristic |
| API key missing | Error with message: "Set ANTHROPIC_API_KEY for --llm anthropic:haiku" |
| Timeout (30s per batch) | Keep original, continue with next batch |

## Estimated Impact

Based on analysis of the ripvec whale session (23.4MB pristine):

| Stage | Chars saved | Cumulative |
|-------|------------|------------|
| Heuristic elision (existing) | ~50k | ~50k |
| LLM exchange summarization | ~200k | ~250k |
| LLM scaffolding stripping | ~150k | ~400k |
| Structural compression (enhanced by distillation) | ~300k | ~700k |
| **Total** | | **~700k chars (~30% of middle zone)** |

The structural compression stage benefits from distillation because the distilled text is denser — char drops, minification, and path shortening operate on higher-value text per byte.
