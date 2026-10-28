#!/usr/bin/env python3
"""Run one strategy on one declared period."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402

STRATEGIES = [
    "debate",
    "debate_no_risk_manager",
    "debate_one_analyst",
    "debate_no_debate",
    "equal_weight_buy_hold",
    "equal_weight_rebalanced",
    "momentum",
    "low_vol",
    "random_agent",
    "single_llm",
    "ensemble",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--period", default="dev", choices=["stress_holdout", "dev", "precutoff_holdout", "clean_test", "forward"])
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--data", default=None, help="price file; default data/dev/prices.parquet")
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--mock-seed", type=int, default=0)
    parser.add_argument("--identified", action="store_true", help="contamination-gap identified mode")
    parser.add_argument("--cost-multiplier", type=float, default=None)
    parser.add_argument("--rebalance-weekday", type=int, default=4)
    parser.add_argument("--frequency", default="weekly", choices=["weekly", "biweekly"])
    parser.add_argument("--universe-fraction", type=float, default=1.0)
    parser.add_argument("--final", action="store_true", help="required for clean_test; human only")
    parser.add_argument("--freeze-path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-llm-calls", type=int, default=1200)
    parser.add_argument("--risk-free-annual", type=float, default=0.0)
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--no-results", action="store_true")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    spec = BacktestSpec(
        strategy=args.strategy,
        period=args.period,
        mode=args.mode,
        data_path=args.data,
        max_decisions=args.max_decisions,
        seed=args.seed,
        mock_seed=args.mock_seed,
        identified=args.identified,
        cost_multiplier=args.cost_multiplier,
        rebalance_weekday=args.rebalance_weekday,
        frequency=args.frequency,
        universe_fraction=args.universe_fraction,
        final=args.final,
        freeze_path=args.freeze_path,
        run_id=args.run_id,
        write_ledger=not args.no_ledger,
        write_trace=not args.no_trace,
        write_results=not args.no_results,
        max_llm_calls=args.max_llm_calls,
        risk_free_annual=args.risk_free_annual,
        results_dir=args.results_dir,
        notes=args.notes,
    )
    result = run_backtest(spec)
    payload = {
        "run_id": result.run_id,
        "strategy": result.strategy,
        "period": result.period,
        "mode": result.mode,
        "model": result.model,
        "metrics": result.metrics,
        "calls": result.calls,
        "provider_calls": result.provider_calls,
        "cost_usd": result.cost_usd,
        "ledger_trial": result.ledger_trial,
        "warnings": result.warnings,
        "flags": result.flags,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
