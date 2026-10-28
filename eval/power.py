#!/usr/bin/env python3
"""Power analysis using baseline/dev data only (never clean-test outcomes)."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402
from core.metrics import equity_returns, minimum_detectable_sharpe  # noqa: E402
from core.periods import load_period_table  # noqa: E402


def _latest_result(pattern: str) -> dict | None:
    files = sorted(glob.glob(str(ROOT / "data" / "results" / pattern)))
    if not files:
        return None
    return json.loads(Path(files[-1]).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="dev")
    parser.add_argument("--data", default=None)
    parser.add_argument("--n-weeks", type=int, default=None)
    parser.add_argument("--weekly-vol", type=float, default=None)
    parser.add_argument("--correlation", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--n-sims", type=int, default=2000)
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    args = parser.parse_args(argv)
    if args.period in {"clean_test", "forward", "embargo"}:
        print(
            "FAIL: power analysis must use dev/baseline data, never a held-out outcome period.",
            file=sys.stderr,
        )
        return 2

    table = load_period_table()
    n_weeks = args.n_weeks or table.clean_test_weeks
    if table.provisional:
        print("WARNING: periods are provisional (model cutoff unverified).", file=sys.stderr)

    base = _latest_result(f"equal_weight_buy_hold__{args.period}__*.json")
    if base is None:
        res = run_backtest(
            BacktestSpec(
                strategy="equal_weight_buy_hold",
                period=args.period,
                mode=args.mode,
                data_path=args.data,
                write_ledger=False,
                write_trace=False,
                write_results=False,
            )
        )
        base = res.to_dict()
    eq = [p["equity"] for p in base["equity"]]
    returns = equity_returns(eq).to_numpy()
    weekly_vol = float(args.weekly_vol if args.weekly_vol is not None else np.std(returns, ddof=1))

    correlation = args.correlation
    if correlation is None:
        debate = _latest_result(f"debate__{args.period}__*.json")
        if debate is not None:
            d_ret = equity_returns([p["equity"] for p in debate["equity"]]).to_numpy()
            n = min(len(returns), len(d_ret))
            correlation = float(np.corrcoef(returns[:n], d_ret[:n])[0, 1])
            if not np.isfinite(correlation):
                correlation = 0.9
        else:
            correlation = 0.9

    mde = minimum_detectable_sharpe(
        n_weeks,
        weekly_volatility=weekly_vol,
        correlation=float(correlation),
        alpha=args.alpha,
        power=args.power,
        n_sims=args.n_sims,
    )
    payload = {
        "period": args.period,
        "n_weeks": n_weeks,
        "weekly_volatility": weekly_vol,
        "correlation_used": correlation,
        "mde": mde,
        "interpretation": (
            "A true annualized Sharpe improvement smaller than mde_sharpe cannot be "
            "distinguished from luck at the pre-registered alpha/power. Clean-test "
            "results must be read with this limitation."
        ),
        "provisional_periods": table.provisional,
    }
    out = ROOT / "data" / "results" / "power_analysis.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
