# agent-trace-viewer

> Zero-config, single-HTML viewer for any LLM agent trace.

Drop a JSONL trace in, get a self-contained HTML page out. Works with **LangChain**, **LangGraph**, **smolagents**, **AgentKit**, and raw **Claude** / **OpenAI** tool-use streams. No service to run, no infrastructure to set up.

```bash
pip install agent-trace-viewer
agent-trace run.jsonl --out report.html
open report.html
```

## Why this exists

If you're building LLM agents, you eventually want to *see* what they did — which tool calls fired, which plans got revised, which steps were slow, what each LLM round cost. The existing options:

- **LangSmith** — paid SaaS, account required.
- **LangFuse** — needs a running server.
- **OpenTelemetry / OpenLLMetry** — heavy, requires a collector + backend.

This is the lightweight option: one Python package, one file out, share it via Slack/Gist/Notion.

## Quick start

### From a JSONL trace file

```bash
agent-trace path/to/run.jsonl --out report.html
```

### From your code

```python
from agent_trace_viewer import render_html

render_html("path/to/run.jsonl", output_path="report.html")
```

### With LangChain

```python
from agent_trace_viewer.adapters.langchain import TraceCallbackHandler

callback = TraceCallbackHandler("run.jsonl")
agent.invoke(input, config={"callbacks": [callback]})
# Then:
#   agent-trace run.jsonl --out report.html
```

*(LangGraph, smolagents, raw Claude/OpenAI adapters — see `docs/adapters.md`.)*

## The Trace v1 schema

The viewer reads one canonical format. Adapters convert from your framework into it.

```json
{"type": "plan",        "ts": 1733000000.1, "duration_ms": 845,  "data": {"plan": "..."}}
{"type": "action",      "ts": 1733000001.5, "duration_ms": 0,    "data": {"tool": "execute_sql", "input": {...}}}
{"type": "observation", "ts": 1733000002.1, "duration_ms": 612,  "data": {"output": "..."}}
{"type": "llm_call",    "ts": 1733000002.7, "duration_ms": 1450, "data": {"model": "claude-sonnet-4-5", "input_tokens": 234, "output_tokens": 88, "cost_usd": 0.0012}}
{"type": "answer",      "ts": 1733000003.4, "duration_ms": 0,    "data": {"text": "..."}}
```

Required: `type`, `ts`. Everything else is optional. Unknown event types render as neutral steps — the viewer never breaks on new types.

Full spec: [`docs/trace-format.md`](docs/trace-format.md).

## Supported sources

| Source | Adapter | Status |
|---|---|---|
| AgentKit | Native (AgentKit already writes Trace v1) | ✅ |
| LangChain | `agent_trace_viewer.adapters.langchain` | 🚧 |
| LangGraph | `agent_trace_viewer.adapters.langgraph` | 🚧 |
| smolagents | `agent_trace_viewer.adapters.smolagents` | 🚧 |
| Raw Anthropic | `agent_trace_viewer.adapters.anthropic` | 🚧 |
| Raw OpenAI | `agent_trace_viewer.adapters.openai` | 🚧 |

(🚧 = MVP target; lands with v0.1.)

## CLI

```
agent-trace <trace.jsonl> [--out OUTPUT.html] [--title TITLE]

Options:
  --out PATH      Output HTML file (default: <trace>.html alongside input)
  --title TEXT    Page title (default: "Agent trace")
  -h, --help      Show this message and exit
```

## Status

**v0.1 — under active development.** The viewer is stable (extracted from AgentKit, well-tested). Adapters are landing this weekend.

## License

Apache 2.0. See [LICENSE](LICENSE).
