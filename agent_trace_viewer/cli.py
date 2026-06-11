"""`agent-trace` CLI — render or validate a JSONL trace."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agent_trace_viewer.schema import validate_trace
from agent_trace_viewer.viewer import render_html


USAGE = """Usage:
  agent-trace <trace.jsonl> [--out OUTPUT.html] [--title TITLE]
  agent-trace --validate <trace.jsonl>

Render a Trace v1 JSONL file as a self-contained HTML page, or validate it
against the Trace v1 schema.

Options:
  --validate      Check the file against Trace v1 (exit 0 = valid, 1 = errors)
  --out PATH      Output HTML file (default: <trace>.html alongside input)
  --title TEXT    Page title (default: "Agent trace")
  -h, --help      Show this message and exit

Examples:
  agent-trace traces/run_1.jsonl
  agent-trace traces/run_1.jsonl --out report.html --title "Nightly QA"
  agent-trace --validate traces/run_1.jsonl
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    trace_path: Path | None = None
    output_path: Path | None = None
    title = "Agent trace"
    validate_only = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--validate":
            validate_only = True
            i += 1
        elif arg == "--out":
            if i + 1 >= len(args):
                print("Error: --out requires a path", file=sys.stderr)
                return 2
            output_path = Path(args[i + 1])
            i += 2
        elif arg == "--title":
            if i + 1 >= len(args):
                print("Error: --title requires text", file=sys.stderr)
                return 2
            title = args[i + 1]
            i += 2
        elif arg.startswith("-"):
            print(f"Error: unknown flag '{arg}'", file=sys.stderr)
            return 2
        else:
            if trace_path is not None:
                print(f"Error: unexpected positional argument '{arg}'", file=sys.stderr)
                return 2
            trace_path = Path(arg)
            i += 1

    if trace_path is None:
        print("Error: provide a trace file path", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 1

    if validate_only:
        return _validate(trace_path)

    try:
        out = render_html(trace_path, output_path=output_path, title=title)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {out}")
    return 0


def _validate(trace_path: Path) -> int:
    if not trace_path.exists():
        print(f"Error: trace file not found: {trace_path}", file=sys.stderr)
        return 1

    events: list[dict] = []
    parse_errors: list[tuple[int, str]] = []
    with trace_path.open() as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError as e:
                parse_errors.append((line_no, f"invalid JSON: {e.msg}"))

    schema_errors = validate_trace(events)

    if not parse_errors and not schema_errors:
        print(f"{trace_path}: OK ({len(events)} events)")
        return 0

    for line_no, msg in parse_errors:
        print(f"{trace_path}:{line_no}: {msg}", file=sys.stderr)
    for err in schema_errors:
        print(f"{trace_path}:{err.line}: {err.message}", file=sys.stderr)
    total = len(parse_errors) + len(schema_errors)
    print(f"\n{total} error(s) in {len(events)} event(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
