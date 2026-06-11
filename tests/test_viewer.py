"""Viewer rendering tests — uses a synthetic Trace v1 stream."""
from __future__ import annotations

import json
from pathlib import Path

from agent_trace_viewer.viewer import render_html, render_events_html


SAMPLE_EVENTS = [
    {"type": "task", "ts": 1700000000.0, "data": {"question": "What was the top product?"}},
    {"type": "plan", "ts": 1700000000.5, "duration_ms": 320,
     "data": {"plan": "list tables -> describe orders -> aggregate revenue"}},
    {"type": "action", "ts": 1700000001.0, "data": {"tool": "list_tables", "input": {}}},
    {"type": "observation", "ts": 1700000001.4, "duration_ms": 380,
     "data": {"output": ["customers", "orders", "products"]}},
    {"type": "llm_call", "ts": 1700000002.0, "duration_ms": 1240,
     "data": {"model": "claude-sonnet-4-5", "input_tokens": 312, "output_tokens": 88}},
    {"type": "answer", "ts": 1700000003.0,
     "data": {"text": "The top product last month was 'Premium Plan'."}},
]


def test_render_html_from_jsonl(tmp_path: Path):
    jsonl = tmp_path / "run.jsonl"
    with jsonl.open("w") as f:
        for e in SAMPLE_EVENTS:
            f.write(json.dumps(e) + "\n")

    html_path = render_html(jsonl)

    assert html_path.exists()
    assert html_path.suffix == ".html"
    content = html_path.read_text()
    assert "Agent trace" in content
    assert "claude-sonnet-4-5" in content
    assert "Premium Plan" in content
    # All event types should color-code into step-<type> classes
    for kind in ("task", "plan", "action", "observation", "llm_call", "answer"):
        assert f"step-{kind}" in content


def test_render_html_with_explicit_output(tmp_path: Path):
    jsonl = tmp_path / "run.jsonl"
    out = tmp_path / "report.html"
    with jsonl.open("w") as f:
        for e in SAMPLE_EVENTS:
            f.write(json.dumps(e) + "\n")

    html_path = render_html(jsonl, output_path=out, title="My QA run")
    assert html_path == out
    assert "My QA run" in out.read_text()


def test_render_html_missing_file_raises(tmp_path: Path):
    try:
        render_html(tmp_path / "nope.jsonl")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_render_html_empty_file_raises(tmp_path: Path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    try:
        render_html(jsonl)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_render_events_html_inline(tmp_path: Path):
    out = tmp_path / "inline.html"
    result = render_events_html(SAMPLE_EVENTS, out, run_id="demo_run")
    assert result == out
    text = out.read_text()
    assert "demo_run" in text
    assert "Premium Plan" in text


def test_legacy_timestamp_key_renders(tmp_path: Path):
    """AgentKit's older traces use 'timestamp' instead of 'ts' — must still render."""
    jsonl = tmp_path / "legacy.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps({"type": "plan", "timestamp": 1700000000.0, "data": {"plan": "x"}}) + "\n")
        f.write(json.dumps({"type": "answer", "timestamp": 1700000001.0, "data": {"text": "y"}}) + "\n")

    html_path = render_html(jsonl)
    content = html_path.read_text()
    assert "step-plan" in content
    assert "step-answer" in content


def test_unknown_event_type_renders_neutral(tmp_path: Path):
    """Unknown event types should render with neutral default styling, not crash."""
    jsonl = tmp_path / "unknown.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps({"type": "custom_event_kind", "ts": 1.0, "data": {"x": 1}}) + "\n")
    html_path = render_html(jsonl)
    assert "step-custom_event_kind" in html_path.read_text()


def test_summary_counts(tmp_path: Path):
    jsonl = tmp_path / "run.jsonl"
    with jsonl.open("w") as f:
        for e in SAMPLE_EVENTS:
            f.write(json.dumps(e) + "\n")
    content = render_html(jsonl).read_text()
    # input_tokens (312) and output_tokens (88) from the llm_call should appear
    assert "312" in content
    assert "88" in content
