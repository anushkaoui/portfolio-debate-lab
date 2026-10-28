"""Append-only trial ledger. Every evaluated configuration is counted."""

from __future__ import annotations

import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import ROOT, git_commit


class Ledger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or ROOT / "ledger.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def count(self) -> int:
        with self.path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())

    def append(
        self,
        *,
        strategy: str,
        period: str,
        model: str,
        mode: str,
        config_hash: str,
        metrics: dict[str, Any],
        cost_usd: float = 0.0,
        calls: int = 0,
        provider_calls: int = 0,
        run_id: str | None = None,
        notes: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        entry = {
            "trial": self.count() + 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "run_id": run_id,
            "strategy": strategy,
            "period": period,
            "model": model,
            "mode": mode,
            "config_hash": config_hash,
            "cost_usd": round(float(cost_usd), 8),
            "calls": int(calls),
            "provider_calls": int(provider_calls),
            "metrics": metrics,
            "notes": notes,
        }
        if extra:
            entry["extra"] = extra
        line = json.dumps(entry, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write(line + "\n")
                fh.flush()
                import os

                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return entry

    def entries(self) -> list[dict]:
        with self.path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def sharpes(self) -> list[float]:
        out = []
        for entry in self.entries():
            value = (entry.get("metrics") or {}).get("sharpe")
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out
