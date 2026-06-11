"""smolagents adapter — convert agent.memory.steps into Trace v1.

smolagents agents accumulate execution history in `agent.memory.steps`. Each
step is one of:

    PlanningStep        — the agent's planning round
    ActionStep          — one tool-use iteration (model output + tool call + observation)
    FinalAnswerStep     — the terminal answer
    TaskStep            — initial task envelope (sometimes present)

This adapter walks that list and emits Trace v1 events.

Usage:

    from smolagents import CodeAgent
    from agent_trace_viewer.adapters.smolagents import agent_to_trace
    from agent_trace_viewer.adapters.anthropic import write_trace_jsonl

    agent = CodeAgent(tools=[...], model=model)
    agent.run("What was our top product last month?")

    events = agent_to_trace(agent)
    write_trace_jsonl(events, "run.jsonl")

Or pass the memory steps directly:

    from agent_trace_viewer.adapters.smolagents import memory_to_trace
    events = memory_to_trace(agent.memory.steps, task=agent.task)

This adapter does NOT import smolagents — it introspects step objects with
`getattr`/`hasattr`, so it stays compatible across smolagents versions.
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterable


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


def _step_kind(step: Any) -> str:
    """Classify a smolagents memory step by its class name."""
    cls = type(step).__name__
    if "Planning" in cls:
        return "planning"
    if "FinalAnswer" in cls:
        return "final"
    if "Action" in cls:
        return "action"
    if "Task" in cls:
        return "task"
    if "SystemPrompt" in cls:
        return "system"
    return "unknown"


def _extract_token_usage(step: Any) -> dict[str, int]:
    """Pull token counts from a step's usage record, regardless of shape."""
    usage = (
        getattr(step, "token_usage", None)
        or getattr(step, "step_token_usage", None)
    )
    if usage is None:
        return {}
    if isinstance(usage, dict):
        in_t = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        out_t = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    else:
        in_t = getattr(usage, "input_tokens", 0) or 0
        out_t = getattr(usage, "output_tokens", 0) or 0
    result: dict[str, int] = {}
    if in_t:
        result["input_tokens"] = int(in_t)
    if out_t:
        result["output_tokens"] = int(out_t)
    return result


def _extract_duration_ms(step: Any) -> float:
    """Duration in ms, from any of the field names smolagents has used."""
    for attr in ("step_duration", "duration", "duration_seconds"):
        v = getattr(step, attr, None)
        if v is not None:
            return float(v) * 1000.0
    start = getattr(step, "start_time", None)
    end = getattr(step, "end_time", None)
    if start is not None and end is not None:
        try:
            return (float(end) - float(start)) * 1000.0
        except (TypeError, ValueError):
            pass
    return 0.0


def _extract_model_thought(step: Any) -> str:
    """The free-text thinking the agent emitted on this step."""
    msg = getattr(step, "model_output_message", None)
    if msg is not None:
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        parts.append(text)
            if parts:
                return " ".join(parts)
    raw = getattr(step, "model_output", None)
    if isinstance(raw, str):
        return raw.strip()
    return ""


def _extract_tool_calls(step: Any) -> list[dict[str, Any]]:
    raw = getattr(step, "tool_calls", None) or []
    out: list[dict[str, Any]] = []
    for tc in raw:
        if isinstance(tc, dict):
            name = tc.get("name") or tc.get("tool_name") or "?"
            args = tc.get("arguments") or tc.get("args") or tc.get("input") or {}
        else:
            name = getattr(tc, "name", None) or getattr(tc, "tool_name", None) or "?"
            args = (
                getattr(tc, "arguments", None)
                or getattr(tc, "args", None)
                or getattr(tc, "input", None)
                or {}
            )
        out.append({"name": name, "arguments": args})
    return out


def _extract_observations(step: Any) -> str:
    obs = getattr(step, "observations", None)
    if obs is None:
        return ""
    if isinstance(obs, list):
        return "\n".join(str(o) for o in obs)
    return str(obs)


def _extract_model_id(step: Any) -> str | None:
    msg = getattr(step, "model_output_message", None)
    if msg is not None:
        for attr in ("model_id", "model_name", "model"):
            v = getattr(msg, attr, None)
            if v is None and isinstance(msg, dict):
                v = msg.get(attr)
            if v:
                return str(v)
    return None


def memory_to_trace(
    steps: Iterable[Any],
    *,
    task: str | None = None,
    model_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Convert smolagents memory steps into Trace v1 events.

    Args:
        steps: Iterable of step objects (typically `agent.memory.steps`).
        task: The user's original task — if provided, emitted as the first event.
              If None, the function will pull it from any TaskStep it finds.
        model_hint: Fallback model name for `llm_call` events when the step
                    doesn't expose one.

    Returns:
        list[dict] — Trace v1 event dicts ready to write as JSONL.
    """
    events: list[dict[str, Any]] = []
    task_emitted = False

    def emit(event_type: str, data: dict[str, Any], duration_ms: float = 0.0) -> None:
        events.append({
            "type": event_type,
            "ts": time.time(),
            "duration_ms": duration_ms,
            "data": data,
        })

    if task:
        emit("task", {"input": _clip(task, 1500)})
        task_emitted = True

    for step in steps:
        kind = _step_kind(step)
        usage = _extract_token_usage(step)
        duration_ms = _extract_duration_ms(step)
        model = _extract_model_id(step) or model_hint

        if kind == "task":
            if task_emitted:
                continue
            t = getattr(step, "task", None) or getattr(step, "input", None)
            if t:
                emit("task", {"input": _clip(t, 1500)})
                task_emitted = True

        elif kind == "system":
            # SystemPromptStep: skip — it's framework boilerplate, not a user-facing step
            continue

        elif kind == "planning":
            if usage or model:
                llm_data: dict[str, Any] = {}
                llm_data.update(usage)
                if model:
                    llm_data["model"] = model
                emit("llm_call", llm_data, duration_ms=duration_ms)
            plan_text = (
                getattr(step, "plan", None)
                or getattr(step, "facts", None)
                or _extract_model_thought(step)
            )
            if plan_text and str(plan_text).strip():
                emit("plan", {"plan": _clip(str(plan_text).strip(), 2000)})

        elif kind == "action":
            if usage or model:
                llm_data = {}
                llm_data.update(usage)
                if model:
                    llm_data["model"] = model
                emit("llm_call", llm_data, duration_ms=duration_ms)

            thought = _extract_model_thought(step)
            if thought:
                emit("plan", {"plan": _clip(thought, 2000)})

            for tc in _extract_tool_calls(step):
                emit("action", {
                    "tool": tc["name"],
                    "input": _clip(tc["arguments"], 2000),
                })

            obs = _extract_observations(step)
            if obs:
                emit("observation", {"output": _clip(obs)})

            error = getattr(step, "error", None)
            if error:
                emit("error", {"source": "action", "error": _clip(str(error), 1000)})

        elif kind == "final":
            answer = (
                getattr(step, "action_output", None)
                or getattr(step, "final_answer", None)
                or _extract_model_thought(step)
            )
            if answer is not None and str(answer).strip():
                emit("answer", {"text": _clip(str(answer), 5000)})

    return events


def agent_to_trace(
    agent: Any,
    *,
    task: str | None = None,
    model_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a smolagents agent's memory to Trace v1 events.

    Best-effort: pulls steps from `agent.memory.steps`, with fallbacks for
    older smolagents that used `agent.logs` instead.
    """
    if task is None:
        task = getattr(agent, "task", None)
    memory = getattr(agent, "memory", None)
    steps = getattr(memory, "steps", None) if memory is not None else None
    if steps is None:
        steps = getattr(agent, "logs", None) or []
    return memory_to_trace(steps, task=task, model_hint=model_hint)
