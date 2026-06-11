"""LangGraph adapter — convert `astream_events()` output into Trace v1.

LangGraph exposes its execution as a stream of `StreamEvent` dicts via
`graph.astream_events(input, version="v2")`. Each dict has the shape:

    {
        "event":      "on_chat_model_end",      # event kind
        "name":       "ChatAnthropic",          # component name
        "run_id":     "...",                    # UUID
        "tags":       [...],
        "metadata":   {"ls_model_name": "claude-sonnet-4-5", ...},
        "data":       {"input": ..., "output": ...},
        "parent_ids": [...],                    # empty list = top-level
    }

This adapter consumes those dicts (sync or async) and produces Trace v1 events.

Usage:

    from agent_trace_viewer.adapters.langgraph import write_trace_from_stream
    from agent_trace_viewer.adapters.anthropic import write_trace_jsonl  # (also exported here)

    async def main():
        events = []
        async for ev in graph.astream_events({"input": "..."}, version="v2"):
            events.append(ev)
        from agent_trace_viewer.adapters.langgraph import events_to_trace
        write_trace_jsonl(events_to_trace(events), "run.jsonl")

Or for the streaming case:

    from agent_trace_viewer.adapters.langgraph import write_trace_from_stream
    await write_trace_from_stream(
        graph.astream_events({"input": "..."}, version="v2"),
        "run.jsonl",
    )

This adapter does NOT import LangGraph — it only inspects dicts that follow the
documented event shape. No extra dependencies required.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, AsyncIterable, Iterable


_TRUNCATE = 5000


def _clip(value: Any, max_len: int = _TRUNCATE) -> str:
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


def _extract_token_usage(output: Any) -> dict[str, Any]:
    """Pull token counts out of a LangChain-shaped LLM response, wherever they live."""
    usage_metadata = getattr(output, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        return usage_metadata

    if isinstance(output, dict):
        for key in ("usage_metadata", "token_usage", "usage"):
            v = output.get(key)
            if isinstance(v, dict):
                return v
        gens = output.get("generations")
        if gens and isinstance(gens, list) and gens and isinstance(gens[0], list) and gens[0]:
            first_gen = gens[0][0]
            if isinstance(first_gen, dict):
                msg = first_gen.get("message")
                if isinstance(msg, dict):
                    um = msg.get("usage_metadata")
                    if isinstance(um, dict):
                        return um

    return {}


def _extract_model(event: dict[str, Any], default: str | None) -> str | None:
    """Find the model name from event metadata, falling back to a hint."""
    metadata = event.get("metadata") or {}
    for key in ("ls_model_name", "model_name", "model"):
        if metadata.get(key):
            return metadata[key]
    if default:
        return default
    name = event.get("name")
    if name and name not in ("ChatModel", "LLM"):
        return name
    return None


def events_to_trace(
    events: Iterable[dict[str, Any]],
    *,
    model_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a sequence of LangGraph StreamEvent dicts into Trace v1 events.

    Args:
        events: Iterable of dicts from `graph.astream_events(input, version="v2")`.
        model_hint: Fallback model name for `llm_call` events when LangGraph
                    doesn't surface it (rare, but happens with custom wrappers).

    Returns:
        list[dict] — Trace v1 event dicts ready to write as JSONL.

    Behaviour:
        - Only the FIRST top-level chain start emits a `task` event.
        - Only the matching top-level chain end emits an `answer` event.
        - Nested chain events (parent_ids non-empty) are ignored for task/answer
          framing, so workflows nested inside agents don't double-emit.
        - `on_chat_model_stream` / `on_llm_stream` chunks are ignored (only the
          terminal `_end` event becomes an `llm_call`).
    """
    out: list[dict[str, Any]] = []
    starts: dict[str, float] = {}
    top_run: str | None = None

    def emit(event_type: str, data: dict[str, Any], duration_ms: float = 0.0) -> None:
        out.append({
            "type": event_type,
            "ts": time.time(),
            "duration_ms": duration_ms,
            "data": data,
        })

    for ev in events:
        kind = ev.get("event", "")
        name = ev.get("name", "")
        run_id = ev.get("run_id")
        data = ev.get("data") or {}
        parent_ids = ev.get("parent_ids") or []
        is_top_level = not parent_ids

        if kind == "on_chain_start" and is_top_level and top_run is None:
            top_run = run_id
            starts[str(run_id)] = time.time()
            input_val = data.get("input")
            emit("task", {"input": _clip(input_val, 1500), "name": name})

        elif kind in ("on_chat_model_start", "on_llm_start"):
            starts[str(run_id)] = time.time()

        elif kind in ("on_chat_model_end", "on_llm_end"):
            duration_ms = (time.time() - starts.pop(str(run_id), time.time())) * 1000
            output = data.get("output")
            llm_data: dict[str, Any] = {}
            usage = _extract_token_usage(output)
            in_t = usage.get("input_tokens") or usage.get("prompt_tokens")
            out_t = usage.get("output_tokens") or usage.get("completion_tokens")
            if in_t:
                llm_data["input_tokens"] = in_t
            if out_t:
                llm_data["output_tokens"] = out_t
            model = _extract_model(ev, model_hint)
            if model:
                llm_data["model"] = model
            emit("llm_call", llm_data, duration_ms=duration_ms)

        elif kind == "on_tool_start":
            starts[str(run_id)] = time.time()
            emit("action", {"tool": name or "?", "input": _clip(data.get("input"), 2000)})

        elif kind == "on_tool_end":
            duration_ms = (time.time() - starts.pop(str(run_id), time.time())) * 1000
            emit("observation", {"output": _clip(data.get("output"))}, duration_ms=duration_ms)

        elif kind in ("on_tool_error", "on_llm_error", "on_chat_model_error"):
            err = data.get("error") if isinstance(data, dict) else None
            emit("error", {"source": kind, "error": _clip(err, 1000)})
            starts.pop(str(run_id), None)

        elif kind == "on_chain_error" and run_id == top_run:
            err = data.get("error") if isinstance(data, dict) else None
            emit("error", {"source": "chain", "error": _clip(err, 1000)})
            starts.pop(str(run_id), None)

        elif kind == "on_chain_end" and run_id == top_run:
            duration_ms = (time.time() - starts.pop(str(run_id), time.time())) * 1000
            emit("answer", {"output": _clip(data.get("output"), 2000)}, duration_ms=duration_ms)

    return out


async def astream_to_trace(
    events: AsyncIterable[dict[str, Any]],
    *,
    model_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Async variant — drain an astream_events() generator into Trace v1 events."""
    collected: list[dict[str, Any]] = []
    async for ev in events:
        collected.append(ev)
    return events_to_trace(collected, model_hint=model_hint)


def write_trace_from_events(
    events: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    model_hint: str | None = None,
) -> Path:
    """Sync convenience: convert + write JSONL in one call."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_events = events_to_trace(events, model_hint=model_hint)
    with output_path.open("w") as f:
        for e in trace_events:
            f.write(json.dumps(e, default=str) + "\n")
    return output_path


async def write_trace_from_stream(
    events: AsyncIterable[dict[str, Any]],
    output_path: str | Path,
    *,
    model_hint: str | None = None,
) -> Path:
    """Async convenience: drain an astream_events() generator + write JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_events = await astream_to_trace(events, model_hint=model_hint)
    with output_path.open("w") as f:
        for e in trace_events:
            f.write(json.dumps(e, default=str) + "\n")
    return output_path
