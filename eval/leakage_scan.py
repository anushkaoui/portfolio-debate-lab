#!/usr/bin/env python3
"""Scan agent transcripts for identifiers, dates or unjustified numeric levels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.anonymize import build_anonymizer_for_universe  # noqa: E402
from core.config import ROOT as CORE_ROOT, load_universe  # noqa: E402
from core.trace import TraceStore  # noqa: E402


def _loads(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _transcript_text(debate) -> str:
    debate = _loads(debate)
    if not debate or not isinstance(debate, dict):
        return ""
    parts = []
    for key in ("responses", "proposals", "critiques", "decision", "trader"):
        value = debate.get(key)
        if value is not None:
            parts.append(json.dumps(value, sort_keys=True))
    for message in debate.get("messages") or []:
        if not isinstance(message, dict):
            continue
        response = message.get("response")
        if response is not None:
            parts.append(json.dumps({"role": message.get("role"), "response": response}, sort_keys=True))
    return "\n".join(parts)


def _numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])\d+\.\d{2,}(?![A-Za-z])", text)


def _context_numbers(value, out: list[float]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _context_numbers(item, out)
    elif isinstance(value, list):
        for item in value:
            _context_numbers(item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out.append(float(value))


def _unmatched_level(token: str, context_numbers: list[float]) -> bool:
    try:
        value = float(token)
    except ValueError:
        return False
    if abs(value) <= 1.0:
        return False
    return not any(abs(value - x) <= max(0.005, 0.01 * abs(value)) for x in context_numbers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(CORE_ROOT / "data" / "traces.sqlite"))
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--out", default=str(CORE_ROOT / "data" / "results" / "leakage_scan.json"))
    args = parser.parse_args(argv)

    db = Path(args.db)
    if not db.exists():
        payload = {"status": "no_trace_db", "db": str(db), "runs": []}
        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0
    store = TraceStore(db)
    anonymizer = build_anonymizer_for_universe(load_universe())
    run_rows = {r["run_id"]: r for r in store.runs()}
    run_ids = args.run_id or list(run_rows)
    report = []
    for run_id in run_ids:
        rows = store.decisions(run_id)
        flags = []
        levels = []
        for row in rows:
            transcript = _transcript_text(row.get("debate_json"))
            context_obj = _loads(row.get("context_json")) or {}
            context_text = json.dumps(context_obj, sort_keys=True)
            context_numbers: list[float] = []
            _context_numbers(context_obj, context_numbers)
            findings = anonymizer.find_identifiers(transcript)
            for finding in findings:
                flags.append({"date": row.get("decision_date"), "type": finding})
            for token in _numeric_tokens(transcript):
                if _unmatched_level(token, context_numbers):
                    levels.append({"date": row.get("decision_date"), "value": token})
        report.append(
            {
                "run_id": run_id,
                "period": (run_rows.get(run_id) or {}).get("period"),
                "decisions": len(rows),
                "identifier_flags": len(flags),
                "identifier_examples": flags[:20],
                "unmatched_numeric_levels": len(levels),
                "numeric_examples": levels[:20],
                "clean": not flags and not levels,
            }
        )
    payload = {
        "db": str(db),
        "runs": report,
        "total_identifier_flags": sum(r["identifier_flags"] for r in report),
        "total_unmatched_numeric_levels": sum(r["unmatched_numeric_levels"] for r in report),
        "note": "Clean-test leakage flags are bugs, not variations.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["total_identifier_flags"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
