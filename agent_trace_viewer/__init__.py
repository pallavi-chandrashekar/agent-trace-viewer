"""agent-trace-viewer — single-file HTML viewer for any LLM agent trace.

Public API:
    render_html(trace_path, output_path=None, title="Agent trace")
    validate_trace(events) -> list[ValidationError]
    TraceEvent (Trace v1 schema dataclass)
"""

from agent_trace_viewer.schema import TraceEvent, ValidationError, validate_trace
from agent_trace_viewer.viewer import render_html, render_events_html

__version__ = "0.1.0"

__all__ = [
    "TraceEvent",
    "ValidationError",
    "validate_trace",
    "render_html",
    "render_events_html",
    "__version__",
]
