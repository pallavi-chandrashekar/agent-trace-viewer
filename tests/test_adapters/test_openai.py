"""OpenAI adapter tests — pure dict translation, no SDK required."""
from __future__ import annotations

from agent_trace_viewer.adapters.openai import messages_to_trace
from agent_trace_viewer.adapters.anthropic import write_trace_jsonl
from agent_trace_viewer.schema import validate_trace


SAMPLE_MESSAGES = [
    {"role": "user", "content": "What's the weather in SF?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "San Francisco"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Sunny, 68°F",
    },
    {
        "role": "assistant",
        "content": "It's sunny and 68°F in San Francisco.",
    },
]


def test_message_history_to_trace_sequence():
    events = messages_to_trace(
        SAMPLE_MESSAGES,
        model="gpt-4o",
        response_usage={"prompt_tokens": 88, "completion_tokens": 24},
    )
    types = [e["type"] for e in events]

    assert types == [
        "task",
        "llm_call",   # first assistant turn (tool call only, no text)
        "action",
        "observation",
        "llm_call",   # second assistant turn
        "plan",       # text content
        "answer",     # final text answer
    ], types

    assert validate_trace(events) == []

    llm_calls = [e for e in events if e["type"] == "llm_call"]
    assert llm_calls[-1]["data"]["input_tokens"] == 88
    assert llm_calls[-1]["data"]["output_tokens"] == 24
    assert llm_calls[-1]["data"]["model"] == "gpt-4o"

    action = next(e for e in events if e["type"] == "action")
    assert action["data"]["tool"] == "get_weather"
    assert "San Francisco" in action["data"]["input"]


def test_handles_unparseable_arguments_gracefully():
    """If the model emits invalid JSON in function arguments, surface them as a string, not crash."""
    msgs = [
        {"role": "user", "content": "X"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "broken", "arguments": "{not-valid-json"},
                }
            ],
        },
    ]
    events = messages_to_trace(msgs)
    action = next(e for e in events if e["type"] == "action")
    assert action["data"]["tool"] == "broken"
    assert "not-valid-json" in action["data"]["input"]


def test_handles_text_only_assistant():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    events = messages_to_trace(msgs)
    types = [e["type"] for e in events]
    assert types == ["task", "llm_call", "plan", "answer"]


def test_renders_via_main_viewer(tmp_path):
    from agent_trace_viewer.viewer import render_html

    events = messages_to_trace(SAMPLE_MESSAGES, model="gpt-4o")
    jsonl = write_trace_jsonl(events, tmp_path / "openai.jsonl")
    html_path = render_html(jsonl)
    content = html_path.read_text()
    assert "step-task" in content
    assert "step-action" in content
    assert "get_weather" in content
