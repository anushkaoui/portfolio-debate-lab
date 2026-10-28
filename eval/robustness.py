#!/usr/bin/env python3
"""Pre-registered robustness checks on the contaminated development period."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402
from core.config import ROOT as CORE_ROOT  # noqa: E402

VARIANTS = {
    "base": {},
    "costs_2x": {"cost_multiplier": 2.0},
    "rebalance_wednesday": {"rebalance_weekday": 2},
    "random_half_universe": {"universe_fraction": 0.5},
    "biweekly": {"frequency": "biweekly"},
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="debate")
    parser.add_argument("--period", default="dev")
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--variants", default="base,costs_2x,rebalance_wednesday,random_half_universe,biweekly")
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--benchmark", default="equal_weight_buy_hold")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--final", action="store_true", help="allow clean_test after an approved freeze")
    parser.add_argument("--freeze-path", default=None)
    parser.add_argument("--out", default=str(CORE_ROOT / "data" / "results" / "robustness.json"))
    args = parser.parse_args(argv)

    rows = []
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name!r}")
        kwargs = VARIANTS[name]
        result = run_backtest(
            BacktestSpec(
                strategy=args.strategy,
                period=args.period,
                mode=args.mode,
                data_path=args.data,
                seed=args.seed,
                max_decisions=args.max_decisions,
                final=args.final,
                freeze_path=args.freeze_path,
                write_ledger=True,
                write_trace=False,
                write_results=True,
                notes=f"Robustness variant {name}; dev only.",
                **kwargs,
            )
        )
        benchmark = None
        if not args.no_benchmark:
            benchmark = run_backtest(
                BacktestSpec(
                    strategy=args.benchmark,
                    period=args.period,
                    mode="mock",
                    data_path=args.data,
                    seed=args.seed,
                    max_decisions=args.max_decisions,
                    final=args.final,
                    freeze_path=args.freeze_path,
                    write_ledger=True,
                    write_trace=False,
                    write_results=True,
                    notes=f"Robustness benchmark {args.benchmark} for variant {name}; dev only.",
                    **kwargs,
                )
            )
        sharpe = result.metrics.get("sharpe")
        benchmark_sharpe = benchmark.metrics.get("sharpe") if benchmark else None
        rows.append(
            {
                "variant": name,
                "run_id": result.run_id,
                "sharpe": sharpe,
                "ann_return": result.metrics.get("annualized_return"),
                "max_drawdown": result.metrics.get("max_drawdown"),
                "cost_drag": result.metrics.get("cost_drag_of_initial"),
                "turnover": result.metrics.get("annualized_turnover"),
                "benchmark": args.benchmark if benchmark else None,
                "benchmark_run_id": benchmark.run_id if benchmark else None,
                "benchmark_sharpe": benchmark_sharpe,
                "excess_sharpe": (
                    float(sharpe) - float(benchmark_sharpe)
                    if sharpe is not None and benchmark_sharpe is not None
                    else None
                ),
                "beats_benchmark": bool(
                    sharpe is not None
                    and benchmark_sharpe is not None
                    and float(sharpe) > float(benchmark_sharpe)
                ),
                "verdict_positive": bool(sharpe is not None and float(sharpe) > 0),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True))
    payload = {
        "strategy": args.strategy,
        "period": args.period,
        "mode": args.mode,
        "rows": rows,
        "caveat": (
            "Dev-only robustness. A configuration that only works in one variant "
            "is fragile and should not be frozen."
        ),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
