"""LLM-assisted compression: classify exchanges, distill, strip scaffolding."""

from .helpers import (
    _REDUCE_TAG_VERSION,
    get_content_blocks,
    get_msg_type,
    get_reduce_tag,
    stamp_reduce_tag,
    was_processed,
)


def _extract_exchange_text(obj):
    """Extract {"role", "text", "tool_name"} dict from a JSONL message object."""
    msg = obj.get("message", {})
    role = msg.get("role", obj.get("type", "unknown"))
    content = msg.get("content", "")
    tool_name = None
    text_parts = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type", "")
            if bt == "text":
                text_parts.append(block.get("text", ""))
            elif bt == "tool_use":
                tool_name = block.get("name")
                text_parts.append(f"[{block.get('name', 'tool')}]")
            elif bt == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str):
                    text_parts.append(inner[:200])

    return {"role": role, "text": "\n".join(text_parts), "tool_name": tool_name}


def _extract_assistant_text(obj):
    """Extract concatenated text from assistant message blocks."""
    if get_msg_type(obj) != "assistant":
        return ""
    blocks = get_content_blocks(obj)
    parts = []
    for b in blocks:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
    return "\n".join(parts)


def _replace_assistant_text(obj, new_text):
    """Replace all text blocks in an assistant message with new_text."""
    msg = obj.get("message", {})
    content = msg.get("content")
    if isinstance(content, list):
        replaced = False
        new_content = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                if not replaced:
                    block["text"] = new_text
                    new_content.append(block)
                    replaced = True
            else:
                new_content.append(block)
        msg["content"] = new_content


def _batched(iterable, n):
    """Yield successive n-sized chunks."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


async def _llm_compression_pass(
    kept_objs, aggr_fn, provider, progress_callback=None, profile="standard"
):
    """Pass 3.6 + 3.7: LLM classification, distillation, and scaffold stripping."""
    import asyncio
    from collections import Counter

    from reduce_session.llm.base import ROUTING_MAP, Route

    stats = {}
    total = len(kept_objs)

    # Identify middle-zone exchanges (aggr > 0.2)
    # Skip messages that were already LLM-processed at the same or higher aggressiveness
    middle = []
    reused_classifications = 0
    for pos, obj in enumerate(kept_objs):
        position = pos / max(total - 1, 1)
        aggr = aggr_fn(position)
        if aggr > 0.2:
            middle.append((pos, obj, aggr))

    if not middle:
        return stats

    # Phase 1: Batched classification with async distillation overlap
    # Reuse cached classifications from _reduce tags where available
    classifications = {}  # pos -> Category
    needs_classification = []  # (pos, obj, aggr) for items needing LLM classification
    from reduce_session.llm.base import Category as _Cat

    for pos, obj, aggr in middle:
        tag = get_reduce_tag(obj)
        if tag and tag.get("v") == _REDUCE_TAG_VERSION and tag.get("cls"):
            _profile_rank = {"gentle": 0, "standard": 1, "aggressive": 2}
            prev_profile = tag.get("profile", "")
            if _profile_rank.get(prev_profile, -1) >= _profile_rank.get(profile, 0):
                # Already classified at same or higher aggressiveness — reuse
                try:
                    classifications[pos] = _Cat(tag["cls"])
                    reused_classifications += 1
                    continue
                except (ValueError, KeyError):
                    pass  # invalid cached classification, re-classify
        needs_classification.append((pos, obj, aggr))

    if reused_classifications:
        stats["llm_classifications_reused"] = reused_classifications

    distill_queue = asyncio.Queue()
    batches = list(_batched(needs_classification, 20)) if needs_classification else []
    total_batches = len(batches)

    # Pre-compute exchange text sizes for sparkline rendering
    exchange_sizes = []

    # Build initial classify_results from cached classifications for sparkline
    cached_classify_results = []
    for pos, obj, aggr in middle:
        text = _extract_assistant_text(obj)
        size = len(text) if text else 0
        cat = classifications.get(pos)
        if cat:
            cached_classify_results.append((cat.value, size))
        else:
            cached_classify_results.append(
                ("", size)
            )  # placeholder for not-yet-classified
    for pos, obj, aggr in middle:
        text = _extract_assistant_text(obj)
        exchange_sizes.append(len(text) if text else 0)

    async def classify_worker():
        # Start with cached classifications for sparkline
        classify_results = list(cached_classify_results)

        # If everything is cached, emit immediately and populate distill queue
        if not needs_classification:
            for pos, obj, aggr in middle:
                cat = classifications.get(pos)
                if cat:
                    route = ROUTING_MAP.get(cat, Route.HEURISTIC)
                    if route == Route.DISTILL and not was_processed(
                        obj, "distilled", profile
                    ):
                        await distill_queue.put((pos, obj, cat))
            if progress_callback:
                progress_callback(
                    {
                        "phase": "classify",
                        "current": len(middle),
                        "total": len(middle),
                        "batch": 0,
                        "total_batches": 0,
                        "classifications": classify_results,
                    }
                )
            await distill_queue.put(None)
            return

        # Build index: middle-list position -> index in classify_results
        mid_pos_to_idx = {pos: idx for idx, (pos, _, _) in enumerate(middle)}

        classified_so_far = reused_classifications
        for batch_num, batch in enumerate(batches, 1):
            exchange_texts = [_extract_exchange_text(obj) for _, obj, _ in batch]
            categories = await provider.classify(exchange_texts)
            for i, ((pos, obj, aggr), cat) in enumerate(zip(batch, categories)):
                classifications[pos] = cat
                stamp_reduce_tag(
                    obj,
                    cls=cat.value,
                    route=ROUTING_MAP.get(cat, Route.HEURISTIC).value,
                    profile=profile,
                )
                route = ROUTING_MAP.get(cat, Route.HEURISTIC)
                if route == Route.DISTILL:
                    if was_processed(obj, "distilled", profile):
                        continue
                    await distill_queue.put((pos, obj, cat))
                # Update the classify_results at the correct position
                idx = mid_pos_to_idx.get(pos)
                if idx is not None and idx < len(classify_results):
                    classify_results[idx] = (cat.value, classify_results[idx][1])
            classified_so_far += len(batch)
            if progress_callback:
                progress_callback(
                    {
                        "phase": "classify",
                        "current": classified_so_far,
                        "total": len(middle),
                        "batch": batch_num,
                        "total_batches": total_batches,
                        "classifications": classify_results,
                    }
                )
        await distill_queue.put(None)  # sentinel

    async def distill_worker():
        distill_count = 0
        chars_saved = 0
        # Count total items in queue (classification is complete, queue is fully populated)
        total_to_distill = distill_queue.qsize() - 1  # subtract sentinel
        if total_to_distill < 0:
            total_to_distill = 0
        processed = 0
        while True:
            item = await distill_queue.get()
            if item is None:
                break
            processed += 1
            pos, obj, cat = item
            text = _extract_assistant_text(obj)
            # Skip short texts — LLM overhead exceeds savings
            reduction_ratio = 0.0
            if text and len(text) > 200:
                original_len = len(text)
                summary = await provider.distill(
                    text, mode="summarize", category=cat.value, profile=profile
                )
                if summary and len(summary) < original_len:
                    _replace_assistant_text(kept_objs[pos], summary)
                    stamp_reduce_tag(kept_objs[pos], distilled=True)
                    distill_count += 1
                    saved = original_len - len(summary)
                    chars_saved += saved
                    reduction_ratio = saved / original_len
            if progress_callback:
                progress_callback(
                    {
                        "phase": "distill",
                        "current": processed,
                        "total": total_to_distill,
                        "chars_saved": chars_saved,
                        "reduction_ratio": reduction_ratio,
                    }
                )
        return distill_count, chars_saved

    # Run classification first (uses classifier model), then distillation
    # (uses distiller model). Sequential avoids loading both models simultaneously
    # and prevents model-switching overhead on local inference.
    await classify_worker()
    distill_count, distill_chars = await distill_worker()

    # Phase 1.5: Distill tool_result content and Agent prompts
    # (only for exchanges classified as DISTILL routes)
    tool_distill_count = 0
    tool_distill_chars = 0

    for pos, obj, aggr in middle:
        # Only process if classified as a DISTILL category
        cat = classifications.get(pos)
        if not cat:
            continue
        route = ROUTING_MAP.get(cat, Route.HEURISTIC)
        if route != Route.DISTILL:
            continue

        t = get_msg_type(obj)
        msg = obj.get("message", {})
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            # Distill tool_result content
            if t == "user" and block.get("type") == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, str) and len(inner) > 200:
                    original_len = len(inner)
                    # Determine tool-specific prompt
                    result_cat = "TOOL_RESULT_DEFAULT"
                    summary = await provider.distill(
                        inner, mode="summarize", category=result_cat, profile=profile
                    )
                    if summary and len(summary) < original_len:
                        block["content"] = summary
                        tool_distill_count += 1
                        tool_distill_chars += original_len - len(summary)

            # Distill Agent prompts
            if t == "assistant" and block.get("type") == "tool_use":
                name = block.get("name", "")
                if name in ("Agent", "agent"):
                    inp = block.get("input", {})
                    if isinstance(inp, dict):
                        prompt_text = inp.get("prompt", "")
                        if isinstance(prompt_text, str) and len(prompt_text) > 200:
                            original_len = len(prompt_text)
                            summary = await provider.distill(
                                prompt_text,
                                mode="summarize",
                                category="AGENT_PROMPT",
                                profile=profile,
                            )
                            if summary and len(summary) < original_len:
                                inp["prompt"] = summary
                                tool_distill_count += 1
                                tool_distill_chars += original_len - len(summary)

        if progress_callback:
            progress_callback(
                {
                    "phase": "distill",
                    "current": tool_distill_count,
                    "total": tool_distill_count,  # we don't know total ahead of time
                    "chars_saved": distill_chars + tool_distill_chars,
                    "reduction_ratio": 0,
                }
            )

    distill_chars += tool_distill_chars
    distill_count += tool_distill_count

    # Phase 2: Scaffolding strip on non-DISTILL assistant text in middle zone
    # DISTILL exchanges already went through summarization — only strip the rest.
    # Also skip short texts (< 200 chars) where LLM overhead exceeds savings.
    distilled_positions = {
        pos
        for pos, _, _ in middle
        if classifications.get(pos)
        and ROUTING_MAP.get(classifications[pos]) == Route.DISTILL
    }

    strip_candidates = []
    total_strip_chars = 0
    for pos, obj, aggr in middle:
        if pos in distilled_positions:
            continue  # already summarized in phase 1
        if was_processed(obj, "scaffold_stripped", profile):
            continue  # already stripped at same+ aggressiveness
        text = _extract_assistant_text(obj)
        if text and len(text) > 200:
            strip_candidates.append((pos, obj, text))
            total_strip_chars += len(text)

    strip_count = 0
    strip_chars_saved = 0
    for idx, (pos, obj, text) in enumerate(strip_candidates, 1):
        original_len = len(text)
        reduction_ratio = 0.0
        stripped = await provider.distill(text, mode="strip_scaffold", profile=profile)
        if stripped and len(stripped) < original_len:
            _replace_assistant_text(kept_objs[pos], stripped)
            stamp_reduce_tag(kept_objs[pos], scaffold_stripped=True)
            strip_count += 1
            saved = original_len - len(stripped)
            strip_chars_saved += saved
            reduction_ratio = saved / original_len
        if progress_callback:
            total_saved = distill_chars + strip_chars_saved
            ratio = total_saved * 100 // max(total_strip_chars + 1, 1)
            progress_callback(
                {
                    "phase": "scaffold",
                    "current": idx,
                    "total": len(strip_candidates),
                    "chars_saved": total_saved,
                    "ratio": ratio,
                    "reduction_ratio": reduction_ratio,
                }
            )

    # Build stats
    route_counts = Counter(
        ROUTING_MAP.get(c, Route.HEURISTIC) for c in classifications.values()
    )

    stats["llm_classified"] = len(classifications)
    stats["llm_classified_keep"] = route_counts.get(Route.KEEP, 0)
    stats["llm_classified_distill"] = route_counts.get(Route.DISTILL, 0)
    stats["llm_classified_heuristic"] = route_counts.get(Route.HEURISTIC, 0)
    stats["llm_distilled"] = distill_count
    stats["llm_scaffold_stripped"] = strip_count
    stats["llm_chars_saved"] = distill_chars + strip_chars_saved
    if tool_distill_count:
        stats["llm_tool_results_distilled"] = tool_distill_count

    return stats
