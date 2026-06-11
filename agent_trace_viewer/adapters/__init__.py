"""Adapters convert framework-specific trace formats into Trace v1 events.

Each adapter exposes either:
    - A callback class (for callback-based frameworks like LangChain)
    - A function that takes the framework's native trace and returns
      a list[TraceEvent] (or writes JSONL directly)

Adapters are imported lazily — installing this package does NOT pull
LangChain/LangGraph/smolagents into your environment. Install the optional
extra for the framework you use:

    pip install "agent-trace-viewer[langchain]"
    pip install "agent-trace-viewer[langgraph]"
    pip install "agent-trace-viewer[smolagents]"

Raw Anthropic/OpenAI adapters work with just the official SDK responses —
no extra deps needed.
"""
