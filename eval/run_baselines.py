#!/usr/bin/env python3
"""Run the frozen baselines on one period and write a comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402
from core.config import ROOT as CORE_ROOT  # noqa: E402

STRATEGIES = [
    "equal_weight_buy_hold",
    "equal_weight_rebalanced",
    "momentum",
    "low_vol",
    "random_agent",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="dev")
    parser.add_argument("--data", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--final", action="store_true", help="allow clean_test after an approved freeze")
    parser.add_argument("--freeze-path", default=None)
    args = parser.parse_args(argv)
    rows = []
    for name in STRATEGIES:
        result = run_backtest(
            BacktestSpec(
                strategy=name,
                period=args.period,
                mode="mock",
                data_path=args.data,
                seed=args.seed,
                final=args.final,
                freeze_path=args.freeze_path,
                write_ledger=not args.no_ledger,
                write_trace=not args.no_trace,
                write_results=True,
            )
        )
        rows.append(
            {
                "strategy": name,
                "run_id": result.run_id,
                "sharpe": result.metrics.get("sharpe"),
                "ann_return": result.metrics.get("annualized_return"),
                "ann_vol": result.metrics.get("annualized_volatility"),
                "max_dd": result.metrics.get("max_drawdown"),
                "turnover": result.metrics.get("annualized_turnover"),
                "cost_drag": result.metrics.get("cost_drag_of_initial"),
            }
        )
    out = CORE_ROOT / "data" / "results" / f"baselines_{args.period}.json"
    out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
