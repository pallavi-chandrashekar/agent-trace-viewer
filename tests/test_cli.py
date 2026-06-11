"""CLI behaviour tests — render mode and --validate mode."""
from __future__ import annotations

import json
from pathlib import Path

from agent_trace_viewer.cli import main


VALID_EVENTS = [
    {"type": "task", "ts": 1.0, "data": {"q": "hi"}},
    {"type": "plan", "ts": 1.5, "duration_ms": 100, "data": {"plan": "say hi"}},
    {"type": "answer", "ts": 2.0, "data": {"text": "hi"}},
]


def _write_jsonl(path: Path, events: list[dict]) -> Path:
    with path.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def test_cli_help_returns_zero(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "agent-trace" in out


def test_cli_render_writes_html(tmp_path, capsys):
    jsonl = _write_jsonl(tmp_path / "run.jsonl", VALID_EVENTS)
    assert main([str(jsonl)]) == 0
    assert (tmp_path / "run.html").exists()
    out = capsys.readouterr().out
    assert "Wrote" in out


def test_cli_render_with_explicit_out(tmp_path):
    jsonl = _write_jsonl(tmp_path / "run.jsonl", VALID_EVENTS)
    out_path = tmp_path / "custom.html"
    assert main([str(jsonl), "--out", str(out_path), "--title", "MyTitle"]) == 0
    assert out_path.exists()
    assert "MyTitle" in out_path.read_text()


def test_cli_validate_clean_file_succeeds(tmp_path, capsys):
    jsonl = _write_jsonl(tmp_path / "run.jsonl", VALID_EVENTS)
    assert main(["--validate", str(jsonl)]) == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "3 events" in out


def test_cli_validate_catches_schema_error(tmp_path, capsys):
    bad = [{"type": "plan"}]  # missing ts
    jsonl = _write_jsonl(tmp_path / "bad.jsonl", bad)
    assert main(["--validate", str(jsonl)]) == 1
    err = capsys.readouterr().err
    assert "ts" in err


def test_cli_validate_catches_invalid_json(tmp_path, capsys):
    jsonl = tmp_path / "broken.jsonl"
    jsonl.write_text('{"type": "plan", "ts": 1.0}\n{this is not JSON\n')
    assert main(["--validate", str(jsonl)]) == 1
    err = capsys.readouterr().err
    assert "invalid JSON" in err


def test_cli_missing_file_returns_1(tmp_path, capsys):
    assert main([str(tmp_path / "nope.jsonl")]) == 1


def test_cli_unknown_flag_returns_2(capsys):
    assert main(["--bogus"]) == 2


def test_cli_no_args_returns_zero_with_usage(capsys):
    # The current behaviour treats no-args as --help and returns 0
    assert main([]) == 0
