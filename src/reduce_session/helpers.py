"""Shared helpers: limit profiles, gradient functions, content-block accessors, metadata.

These are the foundational building blocks used by every other module.
No dependencies on other reduce_session submodules.
"""

# --- Reduction metadata tag ---
# Persisted in JSONL objects to track what processing has been applied.
# Subsequent passes skip already-processed content.

_REDUCE_TAG_VERSION = 1


def get_reduce_tag(obj):
    """Get the _reduce metadata tag from an object, or None."""
    tag = obj.get("_reduce")
    return tag if isinstance(tag, dict) else None


def was_processed(obj, key, profile=None):
    """Check if an object was already processed with the given key at the given profile level."""
    tag = get_reduce_tag(obj)
    if not tag:
        return False
    if tag.get("v") != _REDUCE_TAG_VERSION:
        return False  # different version, reprocess
    if not tag.get(key):
        return False
    # If profile is specified, check if it was at least as aggressive
    if profile:
        _profile_rank = {"gentle": 0, "standard": 1, "aggressive": 2}
        prev = tag.get("profile", "")
        if _profile_rank.get(prev, -1) >= _profile_rank.get(profile, 0):
            return True  # already processed at same or higher aggressiveness
        return False
    return True


def stamp_reduce_tag(obj, **kwargs):
    """Add or update the _reduce metadata tag on an object."""
    tag = obj.get("_reduce", {})
    if not isinstance(tag, dict):
        tag = {}
    tag["v"] = _REDUCE_TAG_VERSION
    tag.update(kwargs)
    obj["_reduce"] = tag


# --- Limit profiles ---

PROFILES = {
    "aggressive": {
        "aggressive": {
            "Bash": 400,
            "Read": 400,
            "Agent": 500,
            "Write": 400,
            "Edit": 300,
            "mcp": 800,
            "default": 400,
            "tur.originalFile": 100,
            "tur.stdout": 400,
            "tur.content": 400,
            "tur.oldString": 200,
            "tur.newString": 200,
            "tur.file": 400,
            "tool_input.Write": 200,
            "tool_input.Edit": 200,
            "tool_input.Agent": 300,
            "tool_input.Bash": 400,
            "thinking": 0,
            "user_text": 800,
        },
        "gentle": {
            "Bash": 3000,
            "Read": 4000,
            "Agent": 6000,
            "Write": 2000,
            "Edit": 1500,
            "mcp": 6000,
            "default": 3000,
            "tur.originalFile": 400,
            "tur.stdout": 3000,
            "tur.content": 2000,
            "tur.oldString": 1500,
            "tur.newString": 1500,
            "tur.file": 2000,
            "tool_input.Write": 2000,
            "tool_input.Edit": 1500,
            "tool_input.Agent": 3000,
            "tool_input.Bash": 2000,
            "thinking": 2000,
            "user_text": 6000,
        },
    },
    "standard": {
        "aggressive": {
            "Bash": 800,
            "Read": 800,
            "Agent": 1000,
            "Write": 600,
            "Edit": 500,
            "mcp": 2000,
            "default": 800,
            "tur.originalFile": 200,
            "tur.stdout": 800,
            "tur.content": 600,
            "tur.oldString": 500,
            "tur.newString": 500,
            "tur.file": 600,
            "tool_input.Write": 600,
            "tool_input.Edit": 500,
            "tool_input.Agent": 800,
            "tool_input.Bash": 600,
            "thinking": 0,
            "user_text": 1500,
        },
        "gentle": {
            "Bash": 4000,
            "Read": 6000,
            "Agent": 8000,
            "Write": 3000,
            "Edit": 2000,
            "mcp": 10000,
            "default": 6000,
            "tur.originalFile": 800,
            "tur.stdout": 4000,
            "tur.content": 3000,
            "tur.oldString": 2000,
            "tur.newString": 2000,
            "tur.file": 3000,
            "tool_input.Write": 3000,
            "tool_input.Edit": 2000,
            "tool_input.Agent": 3000,
            "tool_input.Bash": 3000,
            "thinking": 3000,
            "user_text": 8000,
        },
    },
    "gentle": {
        "aggressive": {
            "Bash": 3000,
            "Read": 4000,
            "Agent": 6000,
            "Write": 2000,
            "Edit": 1500,
            "mcp": 8000,
            "default": 4000,
            "tur.originalFile": 500,
            "tur.stdout": 3000,
            "tur.content": 2000,
            "tur.oldString": 1500,
            "tur.newString": 1500,
            "tur.file": 2000,
            "tool_input.Write": 2000,
            "tool_input.Edit": 1500,
            "tool_input.Agent": 3000,
            "thinking": 1000,
            "user_text": 4000,
        },
        "gentle": {
            "Bash": 12000,
            "Read": 16000,
            "Agent": 20000,
            "Write": 8000,
            "Edit": 6000,
            "mcp": 32000,
            "default": 16000,
            "tur.originalFile": 2000,
            "tur.stdout": 12000,
            "tur.content": 8000,
            "tur.oldString": 6000,
            "tur.newString": 6000,
            "tur.file": 8000,
            "tool_input.Write": 8000,
            "tool_input.Edit": 6000,
            "tool_input.Agent": 12000,
            "thinking": 8000,
            "user_text": 20000,
        },
    },
}

ENVELOPE_FIELDS = {"cwd", "version", "gitBranch", "slug", "userType"}

# --- Gradient functions ---


def make_aggressiveness_fn(cut_pct=10, fade_pct=75):
    """Return a function mapping position [0,1] to aggressiveness [0,1].

    Uses a U-curve: gentle at start and end (high LLM recall zones),
    aggressive in the middle (low recall zone).

    Zones with default cut=10, fade=75:
      [0.00, 0.10]  gentle (0.2)       — start of conversation, high recall
      [0.10, 0.325] ramp up 0.2 → 1.0  — transition to dead zone
      [0.325, 0.425] plateau (1.0)      — middle dead zone, compress hard
      [0.425, 0.75] ramp down 1.0 → 0.2 — transition to recent context
      [0.75, 1.00]  gentle (0.2)        — recent context, high recall
    """
    cut = cut_pct / 100.0  # end of start gentle zone
    fade = fade_pct / 100.0  # start of end gentle zone
    # Plateau spans the middle third of [cut, fade]
    span = fade - cut
    ramp_up_end = cut + span / 3.0
    ramp_down_start = fade - span / 3.0

    def fn(position):
        if position < cut:
            return 0.2  # preserve start
        elif position < ramp_up_end:
            # Ramp from 0.2 to 1.0
            t = (position - cut) / (ramp_up_end - cut)
            return 0.2 + 0.8 * t
        elif position < ramp_down_start:
            return 1.0  # middle dead zone
        elif position < fade:
            # Ramp from 1.0 to 0.2
            t = (position - ramp_down_start) / (fade - ramp_down_start)
            return 1.0 - 0.8 * t
        else:
            return 0.2  # preserve end

    return fn


def blended_limit(key, aggr, aggressive_limits, gentle_limits):
    g = gentle_limits.get(key, gentle_limits["default"])
    a = aggressive_limits.get(key, aggressive_limits["default"])
    return int(g + aggr * (a - g))


# --- Content block helpers ---


def get_content_blocks(msg):
    m = msg.get("message", {})
    content = m.get("content", [])
    return content if isinstance(content, list) else []


def text_of(block):
    for key in ("text", "thinking", "content"):
        val = block.get(key, "")
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list):
            # Handle list-content tool_results: [{"type": "text", "text": "..."}]
            parts = []
            for item in val:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if isinstance(t, str) and t:
                        parts.append(t)
            if parts:
                return "\n".join(parts)
    return ""


def get_msg_type(msg):
    return msg.get("type", "unknown")


# --- Metadata stripping ---

# Fields that are always constant or redundant with filename — safe to strip unconditionally
_ALWAYS_STRIP = {"sessionId", "isSidechain", "entrypoint", "userType"}

# Fields stripped in aggressive mode (not needed for replay)
_AGGRESSIVE_STRIP = {
    "version",
    "requestId",
    "promptId",
    "sourceToolAssistantUUID",
    "slug",
}


def strip_constant_metadata(objs, aggressive=False):
    """Strip redundant constant-value metadata fields from JSONL objects.

    Returns count of fields stripped.
    """
    fields = _ALWAYS_STRIP | (_AGGRESSIVE_STRIP if aggressive else set())
    stripped = 0
    for obj in objs:
        for f in fields:
            if f in obj:
                del obj[f]
                stripped += 1
    return stripped


# --- Line-level filtering ---


def is_droppable_line(obj):
    t = obj.get("type", "")
    if t in ("progress", "file-history-snapshot", "queue-operation", "last-prompt"):
        return t
    if t == "user":
        content = obj.get("message", {}).get("content", "")
        if isinstance(content, str):
            if "<task-notification>" in content:
                return "task_notification"
            if (
                "<local-command-caveat>" in content
                or "<local-command-stdout>" in content
            ):
                return "local_cmd_noise"
            noise_cmds = [
                "/reload-plugins",
                "/plugin",
                "/mcp",
                "/login",
                "/effort",
                "/compact",
            ]
            if "<command-name>" in content:
                for cmd in noise_cmds:
                    if f">{cmd}<" in content:
                        return "local_cmd_noise"
    return None
