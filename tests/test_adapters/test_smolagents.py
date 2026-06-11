"""smolagents adapter tests — using mock step objects, no smolagents install required.

The adapter introspects step objects with getattr, so we just need objects whose
class names match (Planning*, Action*, FinalAnswer*) and which expose the same
attributes as real smolagents steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from agent_trace_viewer.adapters.smolagents import (
    agent_to_trace,
    memory_to_trace,
)
from agent_trace_viewer.schema import validate_trace


# ────────────────────────────────────────────── mocks of smolagents step types
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatMessage:
    content: Any = None
    model_id: str | None = None


@dataclass
class ToolCall:
    name: str
    arguments: Any = None


@dataclass
class TaskStep:
    task: str = ""
    step_duration: float = 0.0
    token_usage: TokenUsage | None = None


@dataclass
class PlanningStep:
    plan: str = ""
    model_output_message: ChatMessage | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    step_duration: float = 0.0


@dataclass
class ActionStep:
    model_output_message: ChatMessage | None = None
    model_output: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: str = ""
    error: Any = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    step_duration: float = 0.0


@dataclass
class FinalAnswerStep:
    action_output: Any = None
    step_duration: float = 0.0


@dataclass
class SystemPromptStep:
    """smolagents emits a SystemPromptStep at the start — we should skip it."""
    system_prompt: str = ""


@dataclass
class MockAgent:
    task: str = ""

    def __post_init__(self):
        self.memory = MockMemory()


@dataclass
class MockMemory:
    steps: list[Any] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────── tests
def _sample_steps() -> list[Any]:
    return [
        SystemPromptStep(system_prompt="You are an agent."),
        PlanningStep(
            plan="First inspect the schema, then aggregate revenue by product.",
            model_output_message=ChatMessage(content="...", model_id="claude-sonnet-4-5"),
            token_usage=TokenUsage(input_tokens=212, output_tokens=68),
            step_duration=1.24,
        ),
        ActionStep(
            model_output_message=ChatMessage(
                content="I'll list tables first to see what's available.",
                model_id="claude-sonnet-4-5",
            ),
            tool_calls=[ToolCall(name="list_tables", arguments={})],
            observations="['customers', 'orders', 'order_items', 'products']",
            token_usage=TokenUsage(input_tokens=298, output_tokens=42),
            step_duration=0.88,
        ),
        ActionStep(
            model_output_message=ChatMessage(
                content="Now I'll aggregate revenue.",
                model_id="claude-sonnet-4-5",
            ),
            tool_calls=[ToolCall(
                name="execute_sql",
                arguments={"query": "SELECT p.name, SUM(oi.quantity * oi.price) AS revenue FROM ..."},
            )],
            observations='[{"name": "Premium Plan", "revenue": 45231.00}]',
            token_usage=TokenUsage(input_tokens=412, output_tokens=88),
            step_duration=1.42,
        ),
        FinalAnswerStep(action_output="The top product was 'Premium Plan' with $45,231 in revenue."),
    ]


def test_full_run_emits_expected_sequence():
    steps = _sample_steps()
    events = memory_to_trace(steps, task="What was our top product last month?")
    types = [e["type"] for e in events]

    assert types == [
        "task",
        "llm_call", "plan",                       # PlanningStep
        "llm_call", "plan", "action", "observation",   # ActionStep 1
        "llm_call", "plan", "action", "observation",   # ActionStep 2
        "answer",                                 # FinalAnswerStep
    ], types
    assert validate_trace(events) == []


def test_token_usage_attached_to_llm_calls():
    steps = _sample_steps()
    events = memory_to_trace(steps, task="X")
    llm_calls = [e for e in events if e["type"] == "llm_call"]
    assert len(llm_calls) == 3
    assert llm_calls[0]["data"]["input_tokens"] == 212
    assert llm_calls[0]["data"]["output_tokens"] == 68
    assert llm_calls[0]["data"]["model"] == "claude-sonnet-4-5"


def test_tool_call_arguments_serialized():
    steps = _sample_steps()
    events = memory_to_trace(steps, task="X")
    actions = [e for e in events if e["type"] == "action"]
    assert actions[0]["data"]["tool"] == "list_tables"
    assert actions[1]["data"]["tool"] == "execute_sql"
    assert "SELECT" in actions[1]["data"]["input"]


def test_system_prompt_step_is_skipped():
    steps = [SystemPromptStep(system_prompt="ignore me"), FinalAnswerStep(action_output="done")]
    events = memory_to_trace(steps, task="hi")
    # No event should reference the system prompt
    types = [e["type"] for e in events]
    assert types == ["task", "answer"]


def test_action_step_with_error_emits_error_event():
    steps = [
        ActionStep(
            model_output="trying broken_tool",
            tool_calls=[ToolCall(name="broken_tool", arguments={})],
            error=RuntimeError("tool exploded"),
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        ),
        FinalAnswerStep(action_output="couldn't finish"),
    ]
    events = memory_to_trace(steps, task="X")
    err = next(e for e in events if e["type"] == "error")
    assert "tool exploded" in err["data"]["error"]


def test_task_step_provides_task_when_no_explicit_task():
    """If we don't pass `task=` and memory has a TaskStep, use that."""
    steps = [
        TaskStep(task="implicit task from TaskStep"),
        FinalAnswerStep(action_output="done"),
    ]
    events = memory_to_trace(steps)
    assert events[0]["type"] == "task"
    assert events[0]["data"]["input"] == "implicit task from TaskStep"


def test_explicit_task_arg_wins_over_task_step():
    """If both explicit task arg and a TaskStep are present, only emit the explicit one."""
    steps = [
        TaskStep(task="memory task"),
        FinalAnswerStep(action_output="done"),
    ]
    events = memory_to_trace(steps, task="explicit task")
    task_events = [e for e in events if e["type"] == "task"]
    assert len(task_events) == 1
    assert task_events[0]["data"]["input"] == "explicit task"


def test_planning_step_without_token_usage_still_emits_plan():
    steps = [
        PlanningStep(plan="figure it out", token_usage=TokenUsage(0, 0)),
        FinalAnswerStep(action_output="done"),
    ]
    events = memory_to_trace(steps, task="X")
    types = [e["type"] for e in events]
    assert "plan" in types
    plan = next(e for e in events if e["type"] == "plan")
    assert plan["data"]["plan"] == "figure it out"


def test_agent_to_trace_pulls_from_memory():
    agent = MockAgent(task="What's the weather?")
    agent.memory.steps = _sample_steps()
    events = agent_to_trace(agent)
    types = [e["type"] for e in events]
    assert types[0] == "task"
    assert types[-1] == "answer"
    assert "What's the weather?" in events[0]["data"]["input"]


def test_agent_to_trace_falls_back_to_logs():
    """Older smolagents used agent.logs instead of agent.memory.steps."""

    class OldAgent:
        task = "legacy task"
        logs = [FinalAnswerStep(action_output="legacy done")]

    events = agent_to_trace(OldAgent())
    assert events[0]["type"] == "task"
    assert events[-1]["type"] == "answer"
    assert events[-1]["data"]["text"] == "legacy done"


def test_model_hint_used_when_step_has_no_model():
    steps = [
        ActionStep(
            model_output="thinking",
            tool_calls=[ToolCall(name="x", arguments={})],
            token_usage=TokenUsage(input_tokens=5, output_tokens=3),
        ),
        FinalAnswerStep(action_output="done"),
    ]
    events = memory_to_trace(steps, task="X", model_hint="my-model")
    llm = next(e for e in events if e["type"] == "llm_call")
    assert llm["data"]["model"] == "my-model"


def test_renders_via_main_viewer(tmp_path):
    from agent_trace_viewer.adapters.anthropic import write_trace_jsonl
    from agent_trace_viewer.viewer import render_html

    events = memory_to_trace(_sample_steps(), task="What was our top product?")
    jsonl = write_trace_jsonl(events, tmp_path / "smol.jsonl")
    html_path = render_html(jsonl)
    content = html_path.read_text()
    assert "step-task" in content
    assert "step-action" in content
    assert "step-llm_call" in content
    assert "Premium Plan" in content
