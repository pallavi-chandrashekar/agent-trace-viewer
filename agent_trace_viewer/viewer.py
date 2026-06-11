"""Render Trace v1 events as a self-contained HTML page.

The output is a single .html file with embedded CSS — no external dependencies,
no JS framework, opens in any browser, shares via Gist/Notion/Slack.

Public API:
    render_html(trace_path, output_path=None, title="Agent trace") -> Path
    render_events_html(events, output_path, run_id=None, title="Agent trace") -> Path

This is the standalone viewer extracted from AgentKit. It accepts the
canonical Trace v1 format (see schema.py) and is intentionally framework-agnostic.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title} · {run_id}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    margin: 0; padding: 32px; background: #fafafa; color: #1a1a1a;
  }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; font-weight: 600; }}
  .meta {{
    color: #666; font-size: 13px; margin-bottom: 24px;
    display: flex; gap: 16px; flex-wrap: wrap;
  }}
  .meta span {{ white-space: nowrap; }}
  .meta b {{ color: #1a1a1a; font-weight: 600; }}
  .summary {{
    background: white; border: 1px solid #e5e5e5; border-radius: 8px;
    padding: 12px 16px; margin-bottom: 16px;
    display: flex; gap: 32px; flex-wrap: wrap;
  }}
  .summary div b {{ display: block; font-size: 18px; }}
  .summary div span {{ color: #888; font-size: 12px; }}
  .timeline {{ display: flex; flex-direction: column; gap: 8px; }}
  .step {{
    background: white; border: 1px solid #e5e5e5; border-radius: 8px;
    padding: 12px 16px; position: relative;
    border-left: 3px solid #ccc;
  }}
  .step-task         {{ border-left-color: #6366f1; }}
  .step-plan         {{ border-left-color: #f59e0b; }}
  .step-action       {{ border-left-color: #3b82f6; }}
  .step-observation  {{ border-left-color: #94a3b8; }}
  .step-reflection   {{ border-left-color: #a855f7; }}
  .step-answer       {{ border-left-color: #22c55e; }}
  .step-llm_call     {{ border-left-color: #10b981; opacity: 0.7; }}
  .step-error        {{ border-left-color: #ef4444; background: #fef2f2; }}
  .step-task_start, .step-task_end {{ border-left-color: #d4d4d4; opacity: 0.6; }}

  .step-header {{
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; color: #666; margin-bottom: 4px;
  }}
  .step-type {{
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
    font-size: 11px;
  }}
  .step-task .step-type         {{ color: #6366f1; }}
  .step-plan .step-type         {{ color: #f59e0b; }}
  .step-action .step-type       {{ color: #3b82f6; }}
  .step-observation .step-type  {{ color: #64748b; }}
  .step-reflection .step-type   {{ color: #a855f7; }}
  .step-answer .step-type       {{ color: #16a34a; }}
  .step-llm_call .step-type     {{ color: #10b981; }}
  .step-error .step-type        {{ color: #dc2626; }}

  .step-body {{
    font-size: 13px; line-height: 1.5; color: #1a1a1a;
    word-wrap: break-word;
  }}
  .step-body pre {{
    background: #f5f5f5; border: 1px solid #e5e5e5; border-radius: 4px;
    padding: 8px 10px; margin: 4px 0 0 0; font-size: 12px;
    overflow-x: auto; white-space: pre-wrap; max-height: 280px;
  }}
  .step-body code {{
    background: #f0f0f0; padding: 1px 4px; border-radius: 3px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px;
  }}
  .step-body details {{ margin-top: 4px; }}
  .step-body summary {{
    cursor: pointer; color: #666; font-size: 12px; user-select: none;
  }}
  .step-body summary:hover {{ color: #1a1a1a; }}
  .step-meta {{ color: #888; font-size: 11px; margin-left: 8px; }}
  .badge {{
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    background: #f0f0f0; color: #555; font-size: 11px;
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>{title}</h1>
  <div class="meta">
    <span>run id <b>{run_id}</b></span>
    <span>started <b>{started_at}</b></span>
    <span>{event_count} events</span>
  </div>
  <div class="summary">
    {summary_html}
  </div>
  <div class="timeline">
    {steps_html}
  </div>
</div>
</body>
</html>
"""


def _format_data(data: dict, max_inline: int = 200) -> str:
    if not data:
        return ""
    parts = []
    for key, value in data.items():
        if isinstance(value, str):
            display = html.escape(value)
            if len(display) > max_inline:
                parts.append(
                    f'<details><summary>{html.escape(key)} ({len(value)} chars)</summary>'
                    f'<pre>{display}</pre></details>'
                )
            else:
                parts.append(f'<div><b>{html.escape(key)}:</b> <code>{display}</code></div>')
        elif isinstance(value, (dict, list)):
            json_str = json.dumps(value, indent=2, default=str)
            display = html.escape(json_str)
            if len(display) > max_inline:
                parts.append(
                    f'<details><summary>{html.escape(key)} ({type(value).__name__})</summary>'
                    f'<pre>{display}</pre></details>'
                )
            else:
                parts.append(f'<div><b>{html.escape(key)}:</b> <code>{display}</code></div>')
        elif isinstance(value, bool):
            parts.append(f'<div><b>{html.escape(key)}:</b> {value}</div>')
        elif value is None:
            continue
        else:
            parts.append(f'<div><b>{html.escape(key)}:</b> {value}</div>')
    return "".join(parts)


def _render_step(event: dict) -> str:
    event_type = event.get("type", "?")
    duration_ms = event.get("duration_ms", 0)
    duration_str = ""
    if duration_ms and duration_ms > 0:
        if duration_ms < 1000:
            duration_str = f'<span class="step-meta">{duration_ms:.0f}ms</span>'
        else:
            duration_str = f'<span class="step-meta">{duration_ms / 1000:.2f}s</span>'

    data = event.get("data", {})
    body_html = _format_data(data) or '<div class="step-meta">(no data)</div>'

    return (
        f'<div class="step step-{html.escape(event_type)}">'
        f'  <div class="step-header">'
        f'    <span class="step-type">{html.escape(event_type)}</span>'
        f'    {duration_str}'
        f'  </div>'
        f'  <div class="step-body">{body_html}</div>'
        f'</div>'
    )


def _render_summary(events: list[dict]) -> str:
    counts: dict[str, int] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    iteration_types = ("plan", "action", "observation", "reflection", "answer")
    for e in events:
        t = e.get("type", "")
        counts[t] = counts.get(t, 0) + 1
        if t == "llm_call":
            d = e.get("data", {}) or {}
            total_input_tokens += d.get("input_tokens", 0) or 0
            total_output_tokens += d.get("output_tokens", 0) or 0

    cards = [
        ("steps", sum(counts.get(t, 0) for t in iteration_types)),
        ("LLM calls", counts.get("llm_call", 0)),
        ("actions", counts.get("action", 0)),
        ("input tokens", total_input_tokens),
        ("output tokens", total_output_tokens),
    ]
    return "".join(
        f'<div><b>{value}</b><span>{label}</span></div>'
        for label, value in cards
    )


def _normalize_ts(event: dict) -> dict:
    """Accept legacy 'timestamp' key alongside Trace v1's 'ts'."""
    if "ts" not in event and "timestamp" in event:
        event = {**event, "ts": event["timestamp"]}
    return event


def _format_started(events: list[dict]) -> str:
    if not events:
        return "?"
    first = events[0]
    ts = first.get("ts", first.get("timestamp"))
    if ts is None:
        return "?"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return "?"


def render_html(
    trace_path: str | Path,
    output_path: str | Path | None = None,
    title: str = "Agent trace",
) -> Path:
    """Read a JSONL trace and write an HTML viewer.

    Args:
        trace_path: Path to a .jsonl file in Trace v1 format.
        output_path: Where to write the HTML. Defaults to <trace>.html alongside input.
        title: Page title.

    Returns:
        Path to the generated HTML file.

    Raises:
        FileNotFoundError: If trace_path doesn't exist.
        ValueError: If the trace file is empty.
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    if output_path is None:
        output_path = trace_path.with_suffix(".html")
    output_path = Path(output_path)

    events: list[dict] = []
    with trace_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(_normalize_ts(json.loads(line)))

    if not events:
        raise ValueError(f"Trace file is empty: {trace_path}")

    run_id = trace_path.stem
    return _write_html(events, output_path, run_id, title)


def render_events_html(
    events: Iterable[dict],
    output_path: str | Path,
    run_id: str | None = None,
    title: str = "Agent trace",
) -> Path:
    """Render an in-memory list of Trace v1 event dicts as HTML.

    Useful for code paths that build a trace programmatically and want to write
    HTML directly without a JSONL roundtrip.
    """
    events_list = [_normalize_ts(dict(e)) for e in events]
    if not events_list:
        raise ValueError("No events to render")
    output_path = Path(output_path)
    return _write_html(events_list, output_path, run_id or "trace", title)


def _write_html(events: list[dict], output_path: Path, run_id: str, title: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = _format_started(events)
    steps_html = "".join(_render_step(e) for e in events)
    summary_html = _render_summary(events)
    content = _HTML_TEMPLATE.format(
        title=html.escape(title),
        run_id=html.escape(run_id),
        started_at=html.escape(started_at),
        event_count=len(events),
        summary_html=summary_html,
        steps_html=steps_html,
    )
    output_path.write_text(content)
    return output_path
