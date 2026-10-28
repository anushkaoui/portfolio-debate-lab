#!/usr/bin/env python3
"""Compare anonymized and identified runs on a contaminated period."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402
from core.config import ROOT as CORE_ROOT, load_config  # noqa: E402
from core.metrics import equity_returns, paired_bootstrap_diff  # noqa: E402


def _returns(result) -> np.ndarray:
    return equity_returns([p["equity"] for p in result.equity]).to_numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="debate")
    parser.add_argument("--period", default="dev")
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--mock-seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--out", default=str(CORE_ROOT / "data" / "results" / "contamination_gap.json"))
    args = parser.parse_args(argv)

    common = dict(
        strategy=args.strategy,
        period=args.period,
        mode=args.mode,
        data_path=args.data,
        seed=args.seed,
        mock_seed=args.mock_seed,
        max_decisions=args.max_decisions,
        write_ledger=True,
        write_trace=False,
        write_results=True,
    )
    anonymized = run_backtest(BacktestSpec(identified=False, **common))
    identified = run_backtest(BacktestSpec(identified=True, **common))
    diff = paired_bootstrap_diff(
        _returns(identified),
        _returns(anonymized),
        n_boot=args.n_boot,
        block_length=args.block_length,
        seed=args.seed,
    )
    exp = load_config("experiment.yaml")
    threshold = float(
        (exp.get("contamination_thresholds") or {}).get(
            "max_abs_contamination_gap_sharpe", 0.50
        )
    )
    payload = {
        "period": args.period,
        "strategy": args.strategy,
        "mode": args.mode,
        "anonymized": {
            "run_id": anonymized.run_id,
            "sharpe": anonymized.metrics.get("sharpe"),
            "total_return": anonymized.metrics.get("total_return"),
        },
        "identified": {
            "run_id": identified.run_id,
            "sharpe": identified.metrics.get("sharpe"),
            "total_return": identified.metrics.get("total_return"),
        },
        "gap_sharpe": diff,
        "threshold_sharpe": threshold,
        "passes_threshold": bool(
            np.isfinite(diff.get("estimate", float("nan")))
            and abs(float(diff["estimate"])) <= threshold
        ),
        "interpretation": (
            "A large gap suggests the model is using memorized identities. "
            "Below threshold does not prove absence of contamination."
        ),
        "provisional": bool(anonymized.metrics.get("provisional_periods")),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
