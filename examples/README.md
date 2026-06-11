# Examples

## Rendered sample traces

Open any of these in a browser to see what `agent-trace-viewer` produces — no install needed:

| Source | Trace (JSONL) | Rendered HTML |
|---|---|---|
| Raw Anthropic (DataAssistant) | [`sample_traces/anthropic.jsonl`](sample_traces/anthropic.jsonl) | [`sample_traces/anthropic.html`](sample_traces/anthropic.html) |
| Raw OpenAI (DataAssistant) | [`sample_traces/openai.jsonl`](sample_traces/openai.jsonl) | [`sample_traces/openai.html`](sample_traces/openai.html) |
| LangChain agent | [`sample_traces/langchain.jsonl`](sample_traces/langchain.jsonl) | [`sample_traces/langchain.html`](sample_traces/langchain.html) |
| LangGraph (MRR analysis) | [`sample_traces/langgraph.jsonl`](sample_traces/langgraph.jsonl) | [`sample_traces/langgraph.html`](sample_traces/langgraph.html) |

## Regenerate samples

```bash
pip install -e ".[langchain,dev]"
python examples/generate_samples.py
```

## Run the CLI on a sample

```bash
agent-trace examples/sample_traces/anthropic.jsonl --out /tmp/anth.html
agent-trace --validate examples/sample_traces/anthropic.jsonl
```
