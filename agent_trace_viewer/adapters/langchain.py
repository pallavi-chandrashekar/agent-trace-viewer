"""LangChain adapter — emit Trace v1 JSONL from a LangChain run.

Usage:

    from agent_trace_viewer.adapters.langchain import TraceCallbackHandler

    handler = TraceCallbackHandler("run.jsonl")
    try:
        agent.invoke({"input": "..."}, config={"callbacks": [handler]})
    finally:
        handler.close()

Then render:

    agent-trace run.jsonl --out report.html

This adapter requires `langchain-core`. Install with:

    pip install "agent-trace-viewer[langchain]"
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as e:  # pragma: no cover - exercised only when extra missing
    raise ImportError(
        "agent_trace_viewer.adapters.langchain requires langchain-core. "
        "Install with: pip install 'agent-trace-viewer[langchain]'"
    ) from e


_TRUNCATE = 5000  # chars; long observation payloads get clipped


def _clip(value: Any, max_len: int = _TRUNCATE) -> str:
    s = str(value)
    if len(s) > max_len:
        return s[:max_len] + f"...(truncated, {len(s)} chars total)"
    return s


class TraceCallbackHandler(BaseCallbackHandler):
    """LangChain BaseCallbackHandler that writes Trace v1 JSONL to a file.

    Mapping from LangChain callbacks to Trace v1 events:

        on_chain_start (top-level only) → task
        on_agent_action                 → plan        (uses action.log)
        on_tool_start                   → action
        on_tool_end                     → observation
        on_tool_error                   → error
        on_llm_end                      → llm_call    (with token usage if present)
        on_llm_error                    → error
        on_chain_end (top-level only)   → answer
        on_agent_finish                 → answer

    Pass to any LangChain agent/chain via `config={"callbacks": [handler]}`.
    """

    def __init__(self, output_path: str | Path):
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w")
        self._llm_starts: dict[UUID, float] = {}
        self._tool_starts: dict[UUID, float] = {}
        self._chain_starts: dict[UUID, float] = {}

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TraceCallbackHandler":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort flush; real users should call .close()
        try:
            self.close()
        except Exception:
            pass

    def _emit(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        event = {
            "type": event_type,
            "ts": time.time(),
            "duration_ms": duration_ms,
            "data": data or {},
        }
        self._file.write(json.dumps(event, default=str) + "\n")
        self._file.flush()

    # ------------------------------------------------------------------ LLM
    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_starts[run_id] = time.time()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_starts[run_id] = time.time()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        duration = (time.time() - self._llm_starts.pop(run_id, time.time())) * 1000
        data: dict[str, Any] = {}
        llm_output = getattr(response, "llm_output", None) or {}
        # token usage can live in a few different shapes
        usage = (
            llm_output.get("token_usage")
            or llm_output.get("usage")
            or {}
        )
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        model = llm_output.get("model_name") or llm_output.get("model")
        if input_tokens:
            data["input_tokens"] = input_tokens
        if output_tokens:
            data["output_tokens"] = output_tokens
        if model:
            data["model"] = model
        self._emit("llm_call", data=data, duration_ms=duration)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._emit("error", data={"source": "llm", "error": str(error)})
        self._llm_starts.pop(run_id, None)

    # ----------------------------------------------------------------- Tool
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_starts[run_id] = time.time()
        tool_name = "?"
        if isinstance(serialized, dict):
            tool_name = serialized.get("name") or serialized.get("id", ["?"])[-1] or "?"
        self._emit("action", data={"tool": tool_name, "input": _clip(input_str)})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        duration = (time.time() - self._tool_starts.pop(run_id, time.time())) * 1000
        self._emit("observation", data={"output": _clip(output)}, duration_ms=duration)

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._emit("error", data={"source": "tool", "error": str(error)})
        self._tool_starts.pop(run_id, None)

    # ---------------------------------------------------------------- Chain
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._chain_starts[run_id] = time.time()
            self._emit("task", data={"input": _clip(inputs, 1000)})

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            duration = (time.time() - self._chain_starts.pop(run_id, time.time())) * 1000
            self._emit("answer", data={"output": _clip(outputs, 2000)}, duration_ms=duration)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._emit("error", data={"source": "chain", "error": str(error)})
            self._chain_starts.pop(run_id, None)

    # ---------------------------------------------------------------- Agent
    def on_agent_action(self, action: Any, *, run_id: UUID, **kwargs: Any) -> None:
        log = getattr(action, "log", None) or ""
        if log.strip():
            self._emit("plan", data={"plan": _clip(log.strip(), 1500)})

    def on_agent_finish(self, finish: Any, *, run_id: UUID, **kwargs: Any) -> None:
        return_values = getattr(finish, "return_values", None) or {}
        self._emit("answer", data={"output": _clip(return_values, 2000)})
