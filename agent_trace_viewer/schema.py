"""Trace v1 schema.

A trace is a sequence of TraceEvents serialized as JSONL (one event per line).

Required fields on every event:
    type: str         The event kind (e.g. "plan", "action", "llm_call"). Free-form;
                      unknown types render as neutral steps in the viewer.
    ts:   float       Unix timestamp (seconds since epoch).

Optional fields:
    duration_ms: float   Wall-clock duration of the event in milliseconds (default 0).
    data:        dict    Event-specific payload (default {}).

Known event types (color-coded in the viewer; any other type is allowed):
    task             — overall task envelope (start)
    plan             — agent's plan
    action           — tool invocation
    observation      — tool result
    reflection       — agent's self-assessment
    answer           — final answer
    llm_call         — a single LLM round-trip
    error            — error / exception

The viewer reads ONLY this format. Adapters convert from LangChain/LangGraph/etc
into this format. Keeping the renderer narrow keeps it stable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


KNOWN_TYPES = frozenset({
    "task", "plan", "action", "observation",
    "reflection", "answer", "llm_call", "error",
    "task_start", "task_end",
})


@dataclass
class TraceEvent:
    """One event in a Trace v1 stream."""
    type: str
    ts: float
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TraceEvent":
        """Lenient parse — fills defaults for missing optional fields, accepts
        legacy 'timestamp' as an alias for 'ts' (AgentKit's earlier name)."""
        if "type" not in raw:
            raise ValueError("TraceEvent: missing required field 'type'")
        ts = raw.get("ts")
        if ts is None:
            ts = raw.get("timestamp")
        if ts is None:
            raise ValueError("TraceEvent: missing required field 'ts' (or legacy 'timestamp')")
        return cls(
            type=str(raw["type"]),
            ts=float(ts),
            duration_ms=float(raw.get("duration_ms", 0.0)),
            data=dict(raw.get("data") or {}),
        )


@dataclass
class ValidationError:
    """A single problem found while validating a trace."""
    line: int      # 1-indexed line number in the source JSONL
    message: str   # human-readable description


def validate_trace(events: Iterable[dict[str, Any]]) -> list[ValidationError]:
    """Check a sequence of raw event dicts against the Trace v1 contract.

    Returns a list of errors. Empty list means the trace is valid.
    Errors are non-fatal — the viewer can still render a partially valid trace
    (it'll skip unrenderable events and surface the count in the summary).
    """
    errors: list[ValidationError] = []
    for i, raw in enumerate(events, start=1):
        if not isinstance(raw, dict):
            errors.append(ValidationError(i, f"event is not an object: {type(raw).__name__}"))
            continue
        if "type" not in raw:
            errors.append(ValidationError(i, "missing required field 'type'"))
        elif not isinstance(raw["type"], str):
            errors.append(ValidationError(i, f"'type' must be a string, got {type(raw['type']).__name__}"))
        if "ts" not in raw and "timestamp" not in raw:
            errors.append(ValidationError(i, "missing required field 'ts' (or legacy 'timestamp')"))
        else:
            ts = raw.get("ts", raw.get("timestamp"))
            if not isinstance(ts, (int, float)):
                errors.append(ValidationError(i, f"'ts' must be a number, got {type(ts).__name__}"))
        if "duration_ms" in raw and not isinstance(raw["duration_ms"], (int, float)):
            errors.append(ValidationError(i, f"'duration_ms' must be a number, got {type(raw['duration_ms']).__name__}"))
        if "data" in raw and raw["data"] is not None and not isinstance(raw["data"], dict):
            errors.append(ValidationError(i, f"'data' must be an object or null, got {type(raw['data']).__name__}"))
    return errors
