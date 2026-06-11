"""Anthropic adapter tests — pure dict translation, no SDK required."""
from __future__ import annotations

from agent_trace_viewer.adapters.anthropic import messages_to_trace, write_trace_jsonl
from agent_trace_viewer.schema import validate_trace


SAMPLE_MESSAGES = [
    {"role": "user", "content": "What's the weather in SF?"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll check the weather tool."},
            {
                "type": "tool_use",
                "id": "tu_1",
                "name": "get_weather",
                "input": {"city": "San Francisco"},
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": "Sunny, 68°F",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "It's sunny and 68°F in San Francisco."}
        ],
    },
]


def test_message_history_to_trace_sequence():
    events = messages_to_trace(
        SAMPLE_MESSAGES,
        model="claude-sonnet-4-5",
        response_usage={"input_tokens": 142, "output_tokens": 38},
    )
    types = [e["type"] for e in events]

    assert types == [
        "task",       # initial user question
        "llm_call",   # first assistant turn
        "plan",       # text block in first assistant turn
        "action",     # tool_use block
        "observation",  # tool_result block in following user turn
        "llm_call",   # second assistant turn
        "plan",       # text block in second assistant turn
        "answer",     # final text answer
    ], types

    assert validate_trace(events) == []

    llm_calls = [e for e in events if e["type"] == "llm_call"]
    # response_usage only attaches to the LAST llm_call
    assert llm_calls[-1]["data"]["input_tokens"] == 142
    assert llm_calls[-1]["data"]["output_tokens"] == 38
    assert llm_calls[-1]["data"]["model"] == "claude-sonnet-4-5"

    action = next(e for e in events if e["type"] == "action")
    assert action["data"]["tool"] == "get_weather"
    assert "San Francisco" in action["data"]["input"]


def test_tool_only_loop_has_no_answer_event():
    """If the final assistant turn is still calling tools, we shouldn't emit `answer`."""
    msgs = [
        {"role": "user", "content": "X"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "foo", "input": {}},
        ]},
    ]
    events = messages_to_trace(msgs)
    assert all(e["type"] != "answer" for e in events)


def test_empty_messages_returns_empty():
    assert messages_to_trace([]) == []


def test_write_trace_jsonl_roundtrip(tmp_path):
    events = messages_to_trace(SAMPLE_MESSAGES, model="claude-sonnet-4-5")
    path = write_trace_jsonl(events, tmp_path / "out.jsonl")
    assert path.exists()
    assert path.read_text().count("\n") == len(events)


def test_renders_via_main_viewer(tmp_path):
    from agent_trace_viewer.viewer import render_html

    events = messages_to_trace(SAMPLE_MESSAGES, model="claude-sonnet-4-5")
    jsonl = write_trace_jsonl(events, tmp_path / "anth.jsonl")
    html_path = render_html(jsonl)
    content = html_path.read_text()
    assert "step-task" in content
    assert "step-action" in content
    assert "step-observation" in content
    assert "get_weather" in content
