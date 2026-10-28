from __future__ import annotations

import json
from datetime import date

import pytest

from core.config import ROOT
from core.freeze import compute_freeze, verify_freeze_lock, write_freeze_lock
from core.ledger import Ledger
from core.trace import TraceStore


def test_ledger_append_and_count(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.count() == 0
    entry = ledger.append(
        strategy="test",
        period="dev",
        model="mock",
        mode="mock",
        config_hash="abc",
        metrics={"sharpe": 0.5},
    )
    assert entry["trial"] == 1
    assert ledger.count() == 1
    assert ledger.sharpes() == [0.5]


def test_trace_roundtrip(tmp_path):
    store = TraceStore(tmp_path / "trace.sqlite")
    store.start_run(
        run_id="r1",
        strategy="test",
        period="dev",
        model="mock",
        mode="mock",
        config_hash="abc",
        git_commit="none",
    )
    store.add_decision(
        run_id="r1",
        decision_date="2023-01-06",
        fill_date="2023-01-09",
        context_hash="h",
        context={"x": 1},
        debate={"y": 2},
        risk={"z": 3},
        fills=[],
        portfolio={},
        flags=[],
        equity=100.0,
    )
    store.finish_run("r1", metrics={"sharpe": 0.2})
    runs = store.runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r1"
    decisions = store.decisions("r1")
    assert len(decisions) == 1
    assert json.loads(decisions[0]["context_json"]) == {"x": 1}


def test_freeze_draft_integrity_and_gate(tmp_path):
    path = write_freeze_lock(tmp_path / "freeze.lock")
    assert path.exists()
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert lock["approved"] is False
    assert lock["combined_sha256"] == compute_freeze()["combined_sha256"]
    ok, reason = verify_freeze_lock(path, require_approved=False)
    assert ok, reason
    ok, reason = verify_freeze_lock(path, require_approved=True)
    assert ok is False
    assert "DRAFT" in reason or "approved" in reason.lower()
