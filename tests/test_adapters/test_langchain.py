"""LangChain adapter tests — drive the callback directly with synthetic LangChain objects."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from agent_trace_viewer.schema import validate_trace

# Skip the whole module if langchain-core isn't installed
pytest.importorskip("langchain_core")

from langchain_core.agents import AgentAction, AgentFinish  # noqa: E402
from langchain_core.outputs import LLMResult  # noqa: E402

from agent_trace_viewer.adapters.langchain import TraceCallbackHandler  # noqa: E402


def _read_events(path: Path) -> list[dict]:
    events = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def test_full_agent_run_emits_expected_event_sequence(tmp_path: Path):
    """Drive the handler through a realistic LangChain run and check the trace."""
    out = tmp_path / "lc.jsonl"
    h = TraceCallbackHandler(out)

    top_run = uuid4()
    h.on_chain_start({"id": ["chain", "AgentExecutor"]}, {"input": "What's 1+1?"}, run_id=top_run)

    h.on_chat_model_start({"id": ["model", "claude"]}, [[{"role": "user", "content": "1+1?"}]], run_id=uuid4())
    h.on_llm_end(
        LLMResult(generations=[[]], llm_output={
            "model_name": "claude-sonnet-4-5",
            "token_usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }),
        run_id=uuid4(),
    )

    h.on_agent_action(
        AgentAction(tool="calculator", tool_input="1+1", log="I should add these numbers"),
        run_id=uuid4(),
    )
    tool_run = uuid4()
    h.on_tool_start({"name": "calculator"}, "1+1", run_id=tool_run)
    h.on_tool_end("2", run_id=tool_run)

    h.on_agent_finish(AgentFinish(return_values={"output": "1+1=2"}, log="done"), run_id=uuid4())
    h.on_chain_end({"output": "1+1=2"}, run_id=top_run)
    h.close()

    events = _read_events(out)
    types = [e["type"] for e in events]

    assert types == ["task", "llm_call", "plan", "action", "observation", "answer", "answer"], types
    assert validate_trace(events) == []

    llm = next(e for e in events if e["type"] == "llm_call")
    assert llm["data"]["input_tokens"] == 12
    assert llm["data"]["output_tokens"] == 6
    assert llm["data"]["model"] == "claude-sonnet-4-5"

    act = next(e for e in events if e["type"] == "action")
    assert act["data"]["tool"] == "calculator"

    obs = next(e for e in events if e["type"] == "observation")
    assert obs["data"]["output"] == "2"


def test_tool_error_emits_error_event(tmp_path: Path):
    out = tmp_path / "err.jsonl"
    h = TraceCallbackHandler(out)

    top_run = uuid4()
    h.on_chain_start({}, {"input": "X"}, run_id=top_run)
    tool_run = uuid4()
    h.on_tool_start({"name": "broken"}, "input", run_id=tool_run)
    h.on_tool_error(RuntimeError("boom"), run_id=tool_run)
    h.on_chain_end({"output": "could not finish"}, run_id=top_run)
    h.close()

    events = _read_events(out)
    assert any(e["type"] == "error" and "boom" in e["data"]["error"] for e in events)


def test_nested_chains_do_not_emit_duplicate_tasks(tmp_path: Path):
    """Only top-level chains (parent_run_id=None) should produce task/answer events."""
    out = tmp_path / "nested.jsonl"
    h = TraceCallbackHandler(out)

    top = uuid4()
    nested = uuid4()
    h.on_chain_start({}, {"input": "X"}, run_id=top)
    h.on_chain_start({}, {"x": 1}, run_id=nested, parent_run_id=top)
    h.on_chain_end({"x": 2}, run_id=nested, parent_run_id=top)
    h.on_chain_end({"output": "done"}, run_id=top)
    h.close()

    events = _read_events(out)
    assert sum(1 for e in events if e["type"] == "task") == 1
    assert sum(1 for e in events if e["type"] == "answer") == 1


def test_renders_via_main_viewer(tmp_path: Path):
    """A LangChain-produced JSONL should render end-to-end through the standard viewer."""
    from agent_trace_viewer.viewer import render_html

    out = tmp_path / "lc.jsonl"
    h = TraceCallbackHandler(out)
    top = uuid4()
    h.on_chain_start({}, {"input": "ping"}, run_id=top)
    h.on_llm_end(
        LLMResult(generations=[[]], llm_output={"model_name": "gpt-4o"}),
        run_id=uuid4(),
    )
    h.on_chain_end({"output": "pong"}, run_id=top)
    h.close()

    html_path = render_html(out)
    content = html_path.read_text()
    assert "step-task" in content
    assert "step-llm_call" in content
    assert "step-answer" in content
