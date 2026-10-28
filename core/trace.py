"""SQLite trace store read by the viewer and used for auditability."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import ROOT


class TraceStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or ROOT / "data" / "traces.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY,
                  created_utc TEXT NOT NULL,
                  strategy TEXT NOT NULL,
                  period TEXT NOT NULL,
                  model TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  config_hash TEXT NOT NULL,
                  metrics_json TEXT,
                  cost_usd REAL NOT NULL DEFAULT 0,
                  calls INTEGER NOT NULL DEFAULT 0,
                  provider_calls INTEGER NOT NULL DEFAULT 0,
                  git_commit TEXT,
                  notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_id TEXT NOT NULL,
                  decision_date TEXT NOT NULL,
                  fill_date TEXT,
                  context_hash TEXT,
                  context_json TEXT,
                  debate_json TEXT,
                  risk_json TEXT,
                  fills_json TEXT,
                  portfolio_json TEXT,
                  flags_json TEXT,
                  equity REAL,
                  FOREIGN KEY(run_id) REFERENCES runs(run_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id, decision_date)"
            )

    def start_run(
        self,
        *,
        run_id: str,
        strategy: str,
        period: str,
        model: str,
        mode: str,
        config_hash: str,
        git_commit: str | None = None,
        notes: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, created_utc, strategy, period, model, mode, config_hash, git_commit, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    strategy,
                    period,
                    model,
                    mode,
                    config_hash,
                    git_commit,
                    notes,
                ),
            )

    def add_decision(
        self,
        *,
        run_id: str,
        decision_date: str,
        fill_date: str | None,
        context_hash: str | None,
        context: Any,
        debate: Any,
        risk: Any,
        fills: Any,
        portfolio: Any,
        flags: Any,
        equity: float | None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions
                (run_id, decision_date, fill_date, context_hash, context_json,
                 debate_json, risk_json, fills_json, portfolio_json, flags_json, equity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    decision_date,
                    fill_date,
                    context_hash,
                    json.dumps(context, sort_keys=True) if context is not None else None,
                    json.dumps(debate, sort_keys=True) if debate is not None else None,
                    json.dumps(risk, sort_keys=True) if risk is not None else None,
                    json.dumps(fills, sort_keys=True) if fills is not None else None,
                    json.dumps(portfolio, sort_keys=True) if portfolio is not None else None,
                    json.dumps(flags, sort_keys=True) if flags is not None else None,
                    equity,
                ),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        metrics: Any,
        cost_usd: float = 0.0,
        calls: int = 0,
        provider_calls: int = 0,
        notes: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs SET metrics_json=?, cost_usd=?, calls=?, provider_calls=?,
                  notes=COALESCE(?, notes)
                WHERE run_id=?
                """,
                (
                    json.dumps(metrics, sort_keys=True) if metrics is not None else None,
                    cost_usd,
                    calls,
                    provider_calls,
                    notes,
                    run_id,
                ),
            )

    def runs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_utc DESC").fetchall()
        return [dict(r) for r in rows]

    def decisions(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE run_id=? ORDER BY decision_date",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]
