"""Regenerate the sample traces in examples/sample_traces/.

Each sample exercises one adapter end-to-end:
- anthropic.jsonl: a 3-turn tool-use loop converted via messages_to_trace
- openai.jsonl:    a 3-turn function-calling loop converted via messages_to_trace
- langchain.jsonl: a synthetic AgentExecutor run driven through TraceCallbackHandler

The synthetic inputs are deterministic; running this script twice produces
identical output (modulo timestamps).
"""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from agent_trace_viewer.adapters.anthropic import (
    messages_to_trace as anthropic_to_trace,
    write_trace_jsonl,
)
from agent_trace_viewer.adapters.openai import messages_to_trace as openai_to_trace
from agent_trace_viewer.adapters.langchain import TraceCallbackHandler
from agent_trace_viewer.viewer import render_html


OUT_DIR = Path(__file__).parent / "sample_traces"


def gen_anthropic() -> None:
    msgs = [
        {"role": "user", "content": "What was our top product last month?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I'll inspect the schema first, then aggregate revenue."},
            {"type": "tool_use", "id": "t1", "name": "list_tables", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1",
             "content": "['customers', 'orders', 'order_items', 'products']"},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Good. Let me join orders ↔ order_items ↔ products and aggregate."},
            {"type": "tool_use", "id": "t2", "name": "execute_sql", "input": {
                "query": "SELECT p.name, SUM(oi.quantity * oi.price) AS revenue "
                         "FROM order_items oi JOIN products p ON p.id = oi.product_id "
                         "JOIN orders o ON o.id = oi.order_id "
                         "WHERE o.order_date >= date('now', '-1 month') "
                         "GROUP BY p.name ORDER BY revenue DESC LIMIT 1"
            }},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2",
             "content": '[{"name": "Premium Plan", "revenue": 45231.00}]'},
        ]},
        {"role": "assistant", "content": [
            {"type": "text",
             "text": "The top product last month was 'Premium Plan' with $45,231 in revenue."},
        ]},
    ]
    events = anthropic_to_trace(
        msgs, model="claude-sonnet-4-5",
        response_usage={"input_tokens": 612, "output_tokens": 187},
    )
    jsonl = write_trace_jsonl(events, OUT_DIR / "anthropic.jsonl")
    render_html(jsonl, title="Anthropic — DataAssistant run")
    print(f"  anthropic.jsonl  {len(events)} events")


def gen_openai() -> None:
    msgs = [
        {"role": "user", "content": "Find the customer with the highest lifetime value."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "list_tables", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "c1",
         "content": "['customers', 'orders', 'order_items']"},
        {"role": "assistant", "content": "Now I'll aggregate order totals per customer.",
         "tool_calls": [{
            "id": "c2", "type": "function",
            "function": {"name": "execute_sql", "arguments":
                '{"query": "SELECT c.name, SUM(oi.quantity * oi.price) AS ltv '
                'FROM customers c JOIN orders o ON o.customer_id=c.id '
                'JOIN order_items oi ON oi.order_id=o.id '
                'GROUP BY c.name ORDER BY ltv DESC LIMIT 1"}'},
        }]},
        {"role": "tool", "tool_call_id": "c2",
         "content": '[{"name": "Acme Corp", "ltv": 128450.00}]'},
        {"role": "assistant",
         "content": "Acme Corp is your highest-LTV customer at $128,450."},
    ]
    events = openai_to_trace(
        msgs, model="gpt-4o",
        response_usage={"prompt_tokens": 488, "completion_tokens": 142},
    )
    jsonl = write_trace_jsonl(events, OUT_DIR / "openai.jsonl")
    render_html(jsonl, title="OpenAI — DataAssistant run")
    print(f"  openai.jsonl     {len(events)} events")


def gen_langchain() -> None:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.outputs import LLMResult

    out = OUT_DIR / "langchain.jsonl"
    h = TraceCallbackHandler(out)
    top = uuid4()
    h.on_chain_start(
        {"id": ["chain", "AgentExecutor"]},
        {"input": "How many active customers do we have?"},
        run_id=top,
    )
    h.on_chat_model_start({}, [[{"role": "user", "content": "..."}]], run_id=uuid4())
    h.on_llm_end(
        LLMResult(generations=[[]], llm_output={
            "model_name": "claude-sonnet-4-5",
            "token_usage": {"prompt_tokens": 412, "completion_tokens": 98},
        }),
        run_id=uuid4(),
    )
    h.on_agent_action(
        AgentAction(tool="list_tables", tool_input="",
                    log="I should check the schema before querying."),
        run_id=uuid4(),
    )
    tr = uuid4()
    h.on_tool_start({"name": "list_tables"}, "", run_id=tr)
    h.on_tool_end("['customers', 'orders', 'subscriptions']", run_id=tr)
    h.on_agent_action(
        AgentAction(tool="execute_sql",
                    tool_input="SELECT COUNT(*) FROM customers WHERE active = 1",
                    log="Now I'll count active customers."),
        run_id=uuid4(),
    )
    tr = uuid4()
    h.on_tool_start({"name": "execute_sql"},
                    "SELECT COUNT(*) FROM customers WHERE active = 1", run_id=tr)
    h.on_tool_end("[(1247,)]", run_id=tr)
    h.on_agent_finish(
        AgentFinish(return_values={"output": "You have 1,247 active customers."}, log="done"),
        run_id=uuid4(),
    )
    h.on_chain_end({"output": "You have 1,247 active customers."}, run_id=top)
    h.close()
    render_html(out, title="LangChain — DataAssistant agent")
    events_count = sum(1 for _ in open(out))
    print(f"  langchain.jsonl  {events_count} events")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing samples to {OUT_DIR}/")
    gen_anthropic()
    gen_openai()
    gen_langchain()


if __name__ == "__main__":
    main()
