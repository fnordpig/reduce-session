"""Token estimation and budget tracking."""

import json

from .helpers import get_content_blocks

CHARS_PER_TOKEN = 3.7


def extract_last_usage(messages):
    """Find the last main-chain assistant message.usage for calibration."""
    for line in reversed(messages):
        obj = json.loads(line) if isinstance(line, str) else line
        if obj.get("type") != "assistant":
            continue
        if obj.get("isSidechain"):
            continue
        usage = obj.get("message", {}).get("usage")
        if usage and isinstance(usage, dict):
            inp = usage.get("input_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_create = usage.get("cache_creation_input_tokens", 0) or 0
            total = inp + cache_read + cache_create
            if total > 0:
                return total
    return None


class TokenBudget:
    """Track estimated context-window token usage by category.

    Only counts text payloads within message.content blocks -- the part that
    fills the context window. If the original file has message.usage data,
    we calibrate our chars/token ratio against the real API count.
    """

    def __init__(self, chars_per_token=CHARS_PER_TOKEN, api_tokens=None):
        self.cpt = chars_per_token
        self.api_tokens = api_tokens  # from last message.usage, for calibration
        self.context = {
            "user_prompts": 0,
            "tool_results": 0,
            "tool_calls": 0,
            "assistant_text": 0,
            "thinking": 0,
            "system": 0,
        }
        self._raw_chars = 0  # total chars before token conversion

    def _tok(self, chars):
        return int(chars / self.cpt)

    def add(self, bucket, chars):
        self._raw_chars += chars
        self.context[bucket] = self.context.get(bucket, 0) + self._tok(chars)

    def _tool_result_text(self, block):
        """Extract only the text payload from a tool_result block."""
        content = block.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for item in content:
                if isinstance(item, dict):
                    total += len(item.get("text", ""))
            return total
        return 0

    def _tool_use_text(self, block):
        """Extract the text payload from a tool_use block's input.

        Only counts string values in the input dict -- the actual content
        the model sees. JSON structure, keys, and UUIDs are not tokens.
        """
        inp = block.get("input", {})
        if not isinstance(inp, dict):
            return 0
        total = 0
        # Count the tool name
        total += len(block.get("name", ""))
        # Count string values in input (file paths, content, prompts, commands)
        for v in inp.values():
            if isinstance(v, str):
                total += len(v)
        return total

    def add_obj(self, obj):
        t = obj.get("type", "")
        if t == "system":
            c = obj.get("message", {}).get("content", "")
            self.add("system", len(c) if isinstance(c, str) else 0)
            return

        blocks = get_content_blocks(obj)
        if t == "user":
            msg = obj.get("message", {})
            content = msg.get("content", "")
            if isinstance(content, str):
                self.add("user_prompts", len(content))
            for b in blocks:
                bt = b.get("type", "")
                if bt == "tool_result":
                    self.add("tool_results", self._tool_result_text(b))
                elif bt == "text":
                    self.add("user_prompts", len(b.get("text", "")))
        elif t == "assistant":
            for b in blocks:
                bt = b.get("type", "")
                if bt == "text":
                    self.add("assistant_text", len(b.get("text", "")))
                elif bt == "tool_use":
                    self.add("tool_calls", self._tool_use_text(b))
                elif bt == "thinking":
                    self.add("thinking", len(b.get("thinking", "")))

    @property
    def context_total(self):
        return sum(self.context.values())

    def report(self, reduced_chars=None):
        """Report token estimates. If reduced_chars given, also estimate reduced tokens."""
        # Calibrate if we have real API data
        calibrated_cpt = self.cpt
        if self.api_tokens and self._raw_chars > 0:
            calibrated_cpt = self._raw_chars / self.api_tokens

        ct = self.context_total
        lines = []

        if self.api_tokens:
            lines.append(
                f"\nToken estimate (calibrated: {calibrated_cpt:.1f} chars/tok from API usage):"
            )
            lines.append(f"  {'last API count':20s} {self.api_tokens:>8,}")
            calibrated_total = int(self._raw_chars / calibrated_cpt)
            lines.append(f"  {'our estimate':20s} {calibrated_total:>8,}")
        else:
            lines.append(
                f"\nToken estimate (heuristic: {self.cpt:.1f} chars/tok, no API data to calibrate):"
            )

        lines.append(f"")
        lines.append(f"  Breakdown by category:")
        for bucket, tokens in sorted(self.context.items(), key=lambda x: -x[1]):
            if tokens > 0:
                pct = tokens / ct * 100 if ct else 0
                lines.append(f"    {bucket:20s} {tokens:>8,} ({pct:4.1f}%)")

        if reduced_chars is not None and self._raw_chars > 0:
            ratio = reduced_chars / self._raw_chars
            if self.api_tokens:
                reduced_tokens = int(self.api_tokens * ratio)
            else:
                reduced_tokens = int(reduced_chars / self.cpt)
            lines.append(f"")
            lines.append(f"  Estimated after reduction: {reduced_tokens:,} tokens")
            if reduced_tokens > 1_000_000:
                lines.append(
                    f"  ** exceeds 1M -- Claude Code will auto-compact on resume **"
                )
            elif (
                self.api_tokens
                and self.api_tokens > 1_000_000
                and reduced_tokens <= 1_000_000
            ):
                lines.append(f"  ** fits in 1M context -- no auto-compact needed **")

        return "\n".join(lines)
