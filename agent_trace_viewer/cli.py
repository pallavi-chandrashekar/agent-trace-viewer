"""`agent-trace` CLI — render a JSONL trace as HTML."""
from __future__ import annotations

import sys
from pathlib import Path

from agent_trace_viewer.viewer import render_html


USAGE = """Usage: agent-trace <trace.jsonl> [--out OUTPUT.html] [--title TITLE]

Render a Trace v1 JSONL file as a self-contained HTML page.

Options:
  --out PATH      Output HTML file (default: <trace>.html alongside input)
  --title TEXT    Page title (default: "Agent trace")
  -h, --help      Show this message and exit

Examples:
  agent-trace traces/run_1.jsonl
  agent-trace traces/run_1.jsonl --out report.html --title "Nightly QA"
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    trace_path: Path | None = None
    output_path: Path | None = None
    title = "Agent trace"

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--out":
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


if __name__ == "__main__":
    sys.exit(main())
