"""LangGraph adapter tests — pure dict translation, no langgraph install required."""
from __future__ import annotations

import asyncio
from pathlib import Path

from agent_trace_viewer.adapters.langgraph import (
    astream_to_trace,
    events_to_trace,
    write_trace_from_events,
    write_trace_from_stream,
)
from agent_trace_viewer.schema import validate_trace


def _stream_event(
    event: str,
    name: str,
    run_id: str,
    *,
    data: dict | None = None,
    metadata: dict | None = None,
    parent_ids: list[str] | None = None,
) -> dict:
    return {
        "event": event,
        "name": name,
        "run_id": run_id,
        "tags": [],
        "metadata": metadata or {},
        "data": data or {},
        "parent_ids": parent_ids or [],
    }


# A realistic LangGraph run: agent decides → tool call → answer
SAMPLE_STREAM = [
    _stream_event("on_chain_start", "LangGraph", "root", data={"input": {"q": "1+1?"}}),
    _stream_event(
        "on_chat_model_start", "ChatAnthropic", "llm1",
        metadata={"ls_model_name": "claude-sonnet-4-5"},
        parent_ids=["root"],
    ),
    _stream_event(
        "on_chat_model_end", "ChatAnthropic", "llm1",
        metadata={"ls_model_name": "claude-sonnet-4-5"},
        data={"output": {"usage_metadata": {"input_tokens": 142, "output_tokens": 38}}},
        parent_ids=["root"],
    ),
    _stream_event(
        "on_tool_start", "calculator", "tool1",
        data={"input": {"expr": "1+1"}},
        parent_ids=["root"],
    ),
    _stream_event(
        "on_tool_end", "calculator", "tool1",
        data={"output": "2"},
        parent_ids=["root"],
    ),
    _stream_event(
        "on_chat_model_start", "ChatAnthropic", "llm2",
        metadata={"ls_model_name": "claude-sonnet-4-5"},
        parent_ids=["root"],
    ),
    _stream_event(
        "on_chat_model_end", "ChatAnthropic", "llm2",
        metadata={"ls_model_name": "claude-sonnet-4-5"},
        data={"output": {"usage_metadata": {"input_tokens": 160, "output_tokens": 14}}},
        parent_ids=["root"],
    ),
    _stream_event("on_chain_end", "LangGraph", "root", data={"output": {"answer": "1+1=2"}}),
]


def test_events_to_trace_emits_expected_sequence():
    events = events_to_trace(SAMPLE_STREAM)
    types = [e["type"] for e in events]

    assert types == ["task", "llm_call", "action", "observation", "llm_call", "answer"], types
    assert validate_trace(events) == []


def test_token_usage_extracted_from_usage_metadata():
    events = events_to_trace(SAMPLE_STREAM)
    llm_calls = [e for e in events if e["type"] == "llm_call"]
    assert len(llm_calls) == 2
    assert llm_calls[0]["data"]["input_tokens"] == 142
    assert llm_calls[0]["data"]["output_tokens"] == 38
    assert llm_calls[0]["data"]["model"] == "claude-sonnet-4-5"
    assert llm_calls[1]["data"]["input_tokens"] == 160
    assert llm_calls[1]["data"]["output_tokens"] == 14


def test_tool_event_carries_name_and_io():
    events = events_to_trace(SAMPLE_STREAM)
    action = next(e for e in events if e["type"] == "action")
    obs = next(e for e in events if e["type"] == "observation")
    assert action["data"]["tool"] == "calculator"
    assert "1+1" in action["data"]["input"]
    assert obs["data"]["output"] == "2"


def test_nested_chain_events_do_not_emit_extra_tasks():
    """parent_ids non-empty → not top-level → no duplicate task/answer events."""
    stream = [
        _stream_event("on_chain_start", "LangGraph", "root", data={"input": "X"}),
        _stream_event(
            "on_chain_start", "InnerNode", "inner", data={"input": "Y"},
            parent_ids=["root"],
        ),
        _stream_event(
            "on_chain_end", "InnerNode", "inner", data={"output": "Y'"},
            parent_ids=["root"],
        ),
        _stream_event("on_chain_end", "LangGraph", "root", data={"output": "X'"}),
    ]
    events = events_to_trace(stream)
    assert sum(1 for e in events if e["type"] == "task") == 1
    assert sum(1 for e in events if e["type"] == "answer") == 1


def test_chain_error_emits_error_event():
    stream = [
        _stream_event("on_chain_start", "LangGraph", "root", data={"input": "X"}),
        _stream_event("on_chain_error", "LangGraph", "root", data={"error": "boom"}),
    ]
    events = events_to_trace(stream)
    err = next(e for e in events if e["type"] == "error")
    assert "boom" in err["data"]["error"]


def test_tool_error_emits_error_event():
    stream = [
        _stream_event("on_chain_start", "LangGraph", "root", data={"input": "X"}),
        _stream_event("on_tool_start", "broken_tool", "t1", data={"input": "x"}, parent_ids=["root"]),
        _stream_event(
            "on_tool_error", "broken_tool", "t1",
            data={"error": "tool failed"}, parent_ids=["root"],
        ),
        _stream_event("on_chain_end", "LangGraph", "root", data={"output": "couldn't finish"}),
    ]
    events = events_to_trace(stream)
    err = next(e for e in events if e["type"] == "error")
    assert "tool failed" in err["data"]["error"]


def test_unknown_events_are_skipped_silently():
    """Stream events we don't care about (on_retriever_*, etc.) shouldn't break anything."""
    stream = [
        _stream_event("on_chain_start", "LangGraph", "root", data={"input": "X"}),
        _stream_event("on_retriever_start", "vectorstore", "r1", parent_ids=["root"]),
        _stream_event(
            "on_retriever_end", "vectorstore", "r1",
            data={"output": ["doc1"]}, parent_ids=["root"],
        ),
        _stream_event("on_chain_end", "LangGraph", "root", data={"output": "X'"}),
    ]
    events = events_to_trace(stream)
    types = [e["type"] for e in events]
    assert types == ["task", "answer"]


def test_model_hint_falls_back_when_metadata_missing():
    stream = [
        _stream_event("on_chain_start", "LangGraph", "root", data={"input": "X"}),
        _stream_event("on_chat_model_start", "ChatModel", "llm1", parent_ids=["root"]),
        _stream_event(
            "on_chat_model_end", "ChatModel", "llm1",
            data={"output": {"usage_metadata": {"input_tokens": 1, "output_tokens": 1}}},
            parent_ids=["root"],
        ),
        _stream_event("on_chain_end", "LangGraph", "root", data={"output": "Y"}),
    ]
    events = events_to_trace(stream, model_hint="my-custom-model")
    llm = next(e for e in events if e["type"] == "llm_call")
    assert llm["data"]["model"] == "my-custom-model"


def test_streaming_chunks_are_ignored():
    """on_chat_model_stream events shouldn't produce duplicate llm_call entries."""
    stream = [
        _stream_event("on_chain_start", "LangGraph", "root", data={"input": "X"}),
        _stream_event("on_chat_model_start", "ChatAnthropic", "llm1", parent_ids=["root"]),
        _stream_event("on_chat_model_stream", "ChatAnthropic", "llm1",
                      data={"chunk": "hello"}, parent_ids=["root"]),
        _stream_event("on_chat_model_stream", "ChatAnthropic", "llm1",
                      data={"chunk": " world"}, parent_ids=["root"]),
        _stream_event(
            "on_chat_model_end", "ChatAnthropic", "llm1",
            data={"output": {"usage_metadata": {"input_tokens": 5, "output_tokens": 2}}},
            parent_ids=["root"],
        ),
        _stream_event("on_chain_end", "LangGraph", "root", data={"output": "hello world"}),
    ]
    events = events_to_trace(stream)
    assert sum(1 for e in events if e["type"] == "llm_call") == 1


def test_write_trace_from_events_roundtrip(tmp_path):
    out = write_trace_from_events(SAMPLE_STREAM, tmp_path / "lg.jsonl")
    assert out.exists()
    assert out.read_text().count("\n") == 6  # task + llm + action + obs + llm + answer


def test_renders_via_main_viewer(tmp_path):
    from agent_trace_viewer.viewer import render_html
    jsonl = write_trace_from_events(SAMPLE_STREAM, tmp_path / "lg.jsonl")
    html_path = render_html(jsonl)
    content = html_path.read_text()
    assert "step-task" in content
    assert "step-action" in content
    assert "step-llm_call" in content
    assert "calculator" in content


def test_async_stream_consumer(tmp_path):
    """astream_to_trace should drain an async iterable and produce the same events."""

    async def fake_stream():
        for ev in SAMPLE_STREAM:
            yield ev

    events = asyncio.run(astream_to_trace(fake_stream()))
    types = [e["type"] for e in events]
    assert types == ["task", "llm_call", "action", "observation", "llm_call", "answer"]


def test_write_trace_from_stream_async(tmp_path):
    """The async convenience writer should produce identical output to the sync one."""

    async def fake_stream():
        for ev in SAMPLE_STREAM:
            yield ev

    out_path = tmp_path / "async.jsonl"
    asyncio.run(write_trace_from_stream(fake_stream(), out_path))
    assert out_path.exists()
    assert out_path.read_text().count("\n") == 6
