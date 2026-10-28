#!/usr/bin/env python3
"""Small pre-registered tuning grid plus walk-forward and sensitivity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402
from core.config import ROOT as CORE_ROOT, load_config, load_yaml  # noqa: E402
from core.metrics import equity_returns, summarize  # noqa: E402


def _fold_sharpes(result, folds: int = 3) -> list[float]:
    points = result.equity
    n = len(points)
    out = []
    for k in range(folds):
        lo = k * n // folds
        hi = (k + 1) * n // folds
        sub = [p["equity"] for p in points[lo:hi]]
        if len(sub) < 4:
            out.append(float("nan"))
        else:
            out.append(float(summarize(sub)["sharpe"]))
    return out


def _run(strategy: str, overrides: dict, *, mode: str, data: str | None, run_id: str, seed: int, notes: str):
    return run_backtest(
        BacktestSpec(
            strategy=strategy,
            period="dev",
            mode=mode,
            data_path=data,
            tunable_overrides=overrides,
            run_id=run_id,
            seed=seed,
            write_ledger=True,
            write_trace=False,
            write_results=True,
            notes=notes,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="debate")
    parser.add_argument("--grid", default=str(ROOT / "config" / "tuning_grid.yaml"))
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--data", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--fold-margin", type=float, default=0.50)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--out", default=str(CORE_ROOT / "data" / "results" / "tuning_report.json"))
    args = parser.parse_args(argv)

    grid = load_yaml(args.grid)
    variants = grid.get("variants") or [{"name": "base", "overrides": {}}]
    base = _run(
        args.strategy,
        {},
        mode=args.mode,
        data=args.data,
        run_id="tune_base",
        seed=args.seed,
        notes="Tuning base config; dev only.",
    )
    baseline = _run(
        "equal_weight_buy_hold",
        {},
        mode="mock",
        data=args.data,
        run_id="tune_equal_weight",
        seed=args.seed,
        notes="Equal-weight reference for tuning gates.",
    )
    baseline_folds = _fold_sharpes(baseline)
    records = []
    for variant in variants:
        overrides = dict(variant.get("overrides") or {})
        run_id = "tune_" + str(variant.get("name", "variant")).replace(" ", "_")
        result = _run(
            args.strategy,
            overrides,
            mode=args.mode,
            data=args.data,
            run_id=run_id,
            seed=args.seed,
            notes=f"Tuning variant {run_id}; dev only.",
        )
        folds = _fold_sharpes(result)
        feasible = all(
            (np.isfinite(f) and np.isfinite(b) and f >= b - args.fold_margin)
            for f, b in zip(folds, baseline_folds)
        ) and float(result.metrics.get("max_drawdown", -1.0)) >= -float(
            load_config("risk_limits.yaml").get("max_drawdown_limit", 0.20)
        )
        records.append(
            {
                "name": variant.get("name"),
                "overrides": overrides,
                "simplicity": len(overrides),
                "sharpe": result.metrics.get("sharpe"),
                "max_drawdown": result.metrics.get("max_drawdown"),
                "fold_sharpes": folds,
                "fold_feasible": bool(feasible),
                "run_id": result.run_id,
            }
        )

    feasible_records = [r for r in records if r["fold_feasible"] and np.isfinite(r["sharpe"])]
    fallback_used = False
    if not feasible_records:
        fallback_used = True
        feasible_records = [r for r in records if np.isfinite(r["sharpe"])]
    best_sharpe = max(r["sharpe"] for r in feasible_records)
    near = [r for r in feasible_records if best_sharpe - r["sharpe"] <= args.tolerance]
    selected = sorted(near, key=lambda r: (r["simplicity"], -r["sharpe"]))[0]
    selected_run = next(r for r in records if r["name"] == selected["name"])

    sensitivity = []
    if args.sensitivity:
        base_tunables = load_config("tunables.yaml")
        sensitivity_values = dict(base_tunables)
        sensitivity_values.update(selected["overrides"])
        for key, value in sorted(sensitivity_values.items()):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key in {"rebalance_weekday", "bootstrap_block_weeks"}:
                continue
            for sign, factor in (("minus", 0.8), ("plus", 1.2)):
                new_value = value * factor
                if isinstance(value, int):
                    new_value = max(1, int(round(new_value)))
                overrides = dict(selected["overrides"])
                overrides[key] = new_value
                run_id = f"tune_sens_{key}_{sign}".replace(".", "_")
                result = _run(
                    args.strategy,
                    overrides,
                    mode=args.mode,
                    data=args.data,
                    run_id=run_id,
                    seed=args.seed,
                    notes=f"Sensitivity {key} {sign} on selected tuning config.",
                )
                improvement = float(result.metrics.get("sharpe", 0)) - float(baseline.metrics.get("sharpe", 0))
                sensitivity.append(
                    {
                        "parameter": key,
                        "value": new_value,
                        "sharpe": result.metrics.get("sharpe"),
                        "beats_equal_weight": improvement > 0,
                        "max_drawdown": result.metrics.get("max_drawdown"),
                        "run_id": result.run_id,
                    }
                )
        signs_by_parameter: dict[str, set[bool]] = {}
        for row in sensitivity:
            signs_by_parameter.setdefault(row["parameter"], set()).add(row["beats_equal_weight"])
        knife_edge = any(len(signs) > 1 for signs in signs_by_parameter.values())
    else:
        knife_edge = None

    payload = {
        "strategy": args.strategy,
        "mode": args.mode,
        "period": "dev",
        "selected": selected,
        "selected_run_id": selected_run["run_id"],
        "best_sharpe": best_sharpe,
        "fallback_used": fallback_used,
        "no_feasible_config": fallback_used,
        "tolerance": args.tolerance,
        "fold_margin": args.fold_margin,
        "baseline_fold_sharpes": baseline_folds,
        "records": records,
        "sensitivity": sensitivity,
        "knife_edge": knife_edge,
        "caveats": [
            "Tuning is dev-only and every variant is in the ledger.",
            "Selected configuration needs human approval before freezing.",
            "If no_feasible_config is true, no grid variant met the fold gate; the reported selection is a debugging fallback and must not be frozen without a human decision.",
            "If knife-edge is true, prefer the simpler default and report instability.",
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("selected", "best_sharpe", "knife_edge", "caveats")}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
