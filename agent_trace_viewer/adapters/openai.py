"""OpenAI adapter — convert an OpenAI chat-completions message history into Trace v1.

Usage with the OpenAI SDK (no extra install required for this adapter):

    from openai import OpenAI
    from agent_trace_viewer.adapters.openai import messages_to_trace
    from agent_trace_viewer.adapters.anthropic import write_trace_jsonl

    client = OpenAI()
    messages = [{"role": "user", "content": "What's the weather?"}]
    # ... run your function-calling loop, accumulating assistant + tool messages ...

    events = messages_to_trace(messages, model="gpt-4o")
    write_trace_jsonl(events, "run.jsonl")

Then:

    agent-trace run.jsonl --out report.html

The function inspects an OpenAI-style message history (with `tool_calls` and
`role: "tool"` reply messages) and emits a Trace v1 event sequence.
"""
from __future__ import annotations

import json
import time
from typing import Any


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
    """Convert an OpenAI chat-completions message history into Trace v1 events.

    Args:
        messages: The list passed to / accumulated through `client.chat.completions.create`.
                  Expected to follow OpenAI's tool-call protocol: assistant messages
                  may include `tool_calls`; tool replies arrive as `role: "tool"`.
        model: Model name to attach to llm_call events.
        started_at: Unix timestamp for the first event. Defaults to time.time().
        response_usage: Optional final usage dict for the last llm_call:
                        {"prompt_tokens": N, "completion_tokens": N}
                        (OpenAI naming — both also normalized to input/output_tokens).

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

        if role == "assistant":
            llm_data: dict[str, Any] = {}
            if model:
                llm_data["model"] = model
            emit("llm_call", llm_data, duration_ms=0)
            last_llm_index = len(events) - 1

            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                emit("plan", {"plan": _stringify(content, 1500)})

            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                func = tc.get("function") or {}
                tool_name = func.get("name") or tc.get("name") or "?"
                args_raw = func.get("arguments") or tc.get("arguments")
                args_parsed: Any
                if isinstance(args_raw, str):
                    try:
                        args_parsed = json.loads(args_raw)
                    except (json.JSONDecodeError, TypeError):
                        args_parsed = args_raw
                else:
                    args_parsed = args_raw
                emit("action", {"tool": tool_name, "input": _stringify(args_parsed, 2000)})

        elif role == "tool":
            output = msg.get("content")
            emit("observation", {"output": _stringify(output, 5000)})

    if response_usage and last_llm_index is not None:
        last = events[last_llm_index]["data"]
        in_t = response_usage.get("prompt_tokens") or response_usage.get("input_tokens")
        out_t = response_usage.get("completion_tokens") or response_usage.get("output_tokens")
        if in_t:
            last["input_tokens"] = in_t
        if out_t:
            last["output_tokens"] = out_t

    last_assistant = next(
        (m for m in reversed(messages) if m.get("role") == "assistant"), None
    )
    if last_assistant is not None:
        content = last_assistant.get("content")
        tool_calls = last_assistant.get("tool_calls") or []
        if isinstance(content, str) and content.strip() and not tool_calls:
            emit("answer", {"text": _stringify(content, 5000)})

    return events
