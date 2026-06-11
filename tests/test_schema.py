"""Schema tests for Trace v1."""
from __future__ import annotations

from agent_trace_viewer.schema import TraceEvent, validate_trace


def test_trace_event_minimal():
    e = TraceEvent(type="plan", ts=1733000000.0)
    assert e.duration_ms == 0.0
    assert e.data == {}


def test_trace_event_to_dict_roundtrip():
    e = TraceEvent(type="action", ts=1.5, duration_ms=12.5, data={"tool": "x"})
    raw = e.to_dict()
    restored = TraceEvent.from_dict(raw)
    assert restored == e


def test_from_dict_accepts_legacy_timestamp():
    raw = {"type": "plan", "timestamp": 100.0, "data": {"k": "v"}}
    e = TraceEvent.from_dict(raw)
    assert e.ts == 100.0
    assert e.data == {"k": "v"}


def test_from_dict_requires_type():
    raw = {"ts": 1.0}
    try:
        TraceEvent.from_dict(raw)
    except ValueError as exc:
        assert "type" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing type")


def test_from_dict_requires_ts():
    raw = {"type": "plan"}
    try:
        TraceEvent.from_dict(raw)
    except ValueError as exc:
        assert "ts" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing ts")


def test_validate_trace_clean_pass():
    events = [
        {"type": "plan", "ts": 1.0, "data": {}},
        {"type": "action", "ts": 2.0, "duration_ms": 10},
    ]
    errors = validate_trace(events)
    assert errors == []


def test_validate_trace_legacy_timestamp_ok():
    events = [{"type": "plan", "timestamp": 1.0}]
    errors = validate_trace(events)
    assert errors == []


def test_validate_trace_catches_missing_type():
    events = [{"ts": 1.0}]
    errors = validate_trace(events)
    assert len(errors) == 1
    assert errors[0].line == 1
    assert "type" in errors[0].message


def test_validate_trace_catches_bad_data_type():
    events = [{"type": "plan", "ts": 1.0, "data": "not a dict"}]
    errors = validate_trace(events)
    assert any("data" in e.message for e in errors)


def test_validate_trace_catches_bad_ts():
    events = [{"type": "plan", "ts": "not a number"}]
    errors = validate_trace(events)
    assert any("ts" in e.message and "number" in e.message for e in errors)
