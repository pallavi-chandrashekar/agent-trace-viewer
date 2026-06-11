"""Anthropic adapter — convert a raw Anthropic message stream into Trace v1.

Usage with the Anthropic SDK (no extra install required for this adapter):

    from anthropic import Anthropic
    from agent_trace_viewer.adapters.anthropic import messages_to_trace

    client = Anthropic()
    messages = [{"role": "user", "content": "What's the weather?"}]
    response = client.messages.create(
        model="claude-sonnet-4-5",
        messages=messages,
        tools=[...],
        max_tokens=1024,
    )
    # ... run your tool-use loop, accumulating message turns ...

    events = messages_to_trace(messages, model="claude-sonnet-4-5")
    write_trace_jsonl(events, "run.jsonl")

Then:

    agent-trace run.jsonl --out report.html

The function inspects an Anthropic-style message history (with `tool_use` and
`tool_result` content blocks) and emits a Trace v1 event sequence.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable


def _now() -> float:
    return time.time()


def _stringify(value: Any, max_len: int = 5000) -> str:
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str)
        except (TypeError, ValueError):
            s = str(value)
    if len(s) > max_len:
        return s[:max_len] + f"...(truncated, {len(s)} chars total)"
    return s


def messages_to_trace(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    started_at: float | None = None,
    response_usage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert an Anthropic message history into Trace v1 events.

    Args:
        messages: The list passed to / returned by `client.messages.create`.
                  Expected to follow Anthropic's tool-use protocol — assistant
                  messages may contain `tool_use` blocks; user messages may
                  contain `tool_result` blocks.
        model: Model name to attach to any `llm_call` events (best-effort).
        started_at: Unix timestamp for the first event. Defaults to time.time().
        response_usage: Optional final usage dict to attach to the last llm_call:
                        {"input_tokens": N, "output_tokens": N}.

    Returns:
        list[dict] — Trace v1 event dicts ready to write as JSONL.
    """
    ts = started_at if started_at is not None else _now()
    events: list[dict[str, Any]] = []

    def emit(event_type: str, data: dict[str, Any], duration_ms: float = 0.0) -> None:
        nonlocal ts
        events.append({"type": event_type, "ts": ts, "duration_ms": duration_ms, "data": data})
        ts += max(duration_ms / 1000.0, 0.001)

    first_user = next((m for m in messages if m.get("role") == "user"), None)
    if first_user is not None:
        emit("task", {"input": _stringify(first_user.get("content"), 1000)})

    last_llm_index: int | None = None
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant":
            llm_data: dict[str, Any] = {}
            if model:
                llm_data["model"] = model
            emit("llm_call", llm_data, duration_ms=0)
            last_llm_index = len(events) - 1

            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                btype = block.get("type") if isinstance(block, dict) else None
                if btype == "text":
                    text = (block.get("text") or "").strip() if isinstance(block, dict) else ""
                    if text:
                        emit("plan", {"plan": _stringify(text, 1500)})
                elif btype == "tool_use":
                    emit("action", {
                        "tool": block.get("name", "?"),
                        "input": _stringify(block.get("input"), 2000),
                    })

        elif role == "user":
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    emit("observation", {
                        "output": _stringify(block.get("content"), 5000),
                        "is_error": bool(block.get("is_error", False)),
                    })

    if response_usage and last_llm_index is not None:
        last = events[last_llm_index]["data"]
        if "input_tokens" in response_usage:
            last["input_tokens"] = response_usage["input_tokens"]
        if "output_tokens" in response_usage:
            last["output_tokens"] = response_usage["output_tokens"]

    last_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    if last_assistant is not None:
        text_parts: list[str] = []
        c = last_assistant.get("content")
        blocks = c if isinstance(c, list) else [{"type": "text", "text": c}]
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                t = (block.get("text") or "").strip()
                if t:
                    text_parts.append(t)
        if text_parts and not any(b.get("type") == "tool_use" for b in blocks if isinstance(b, dict)):
            emit("answer", {"text": _stringify(" ".join(text_parts), 5000)})

    return events


def write_trace_jsonl(
    events: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a list of Trace v1 event dicts to a JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for e in events:
            f.write(json.dumps(e, default=str) + "\n")
    return output_path
