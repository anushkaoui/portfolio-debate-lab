#!/usr/bin/env python3
"""Compare saved backtest results, with CIs, null percentile, DSR and power."""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_config  # noqa: E402
from core.ledger import Ledger  # noqa: E402
from core.periods import load_period_table  # noqa: E402
from core.metrics import (  # noqa: E402
    bootstrap_ci,
    deflated_sharpe_ratio,
    equity_returns,
    paired_bootstrap_diff,
    random_agent_percentile,
)


def _load_results(directory: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith(("null_", "power_", "probes_", "leakage_", "contamination_", "analysis_", "tuning_")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "run_id" not in data or "metrics" not in data or "equity" not in data:
            continue
        out[data["run_id"]] = data
    return out


def _select_canonical(results: dict[str, dict], *, include_identified: bool) -> dict[str, dict]:
    """Keep one representative run per strategy (base config, prefer live)."""
    groups: dict[str, list[tuple[str, dict]]] = {}
    for run_id, result in results.items():
        identified = bool(result.get("identified") or result.get("metrics", {}).get("identified"))
        if identified and not include_identified:
            continue
        groups.setdefault(result["strategy"], []).append((run_id, result))

    selected: dict[str, dict] = {}
    for strategy, items in groups.items():
        def score(item: tuple[str, dict]) -> tuple:
            run_id, result = item
            spec = result.get("spec") or {}
            canonical = (
                str(spec.get("frequency", "weekly")) == "weekly"
                and float(spec.get("cost_multiplier") or 1.0) == 1.0
                and float(spec.get("universe_fraction") or 1.0) == 1.0
                and int(spec.get("rebalance_weekday", 4)) == 4
                and spec.get("max_decisions") is None
                and not spec.get("start_date")
                and not spec.get("end_date")
                and not spec.get("tunable_overrides")
            )
            return (
                canonical,
                str(result.get("mode")) == "live",
                bool(spec),
                run_id,
            )

        chosen = max(items, key=score)[1]
        selected[str(chosen["run_id"])] = chosen
    return selected


def _returns(result: dict) -> np.ndarray:
    return equity_returns([float(p["equity"]) for p in result["equity"]]).to_numpy()


def _block_bootstrap_stat(
    values: np.ndarray,
    stat,
    *,
    n_boot: int,
    block_length: int,
    seed: int,
) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"value": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_length))
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - block_length + 1), size=n_blocks)
        sample = np.concatenate([values[s : s + block_length] for s in starts])[:n]
        estimates[i] = stat(sample)
    return {
        "value": float(stat(values)),
        "low": float(np.nanquantile(estimates, 0.025)),
        "high": float(np.nanquantile(estimates, 0.975)),
    }


def _max_drawdown_from_returns(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + np.asarray(values, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def _fmt(x: float) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "data" / "results"))
    parser.add_argument("--period", default="dev")
    parser.add_argument("--run-id", action="append", default=None)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--out", default=None)
    parser.add_argument("--include-identified", action="store_true")
    args = parser.parse_args(argv)

    results = _load_results(Path(args.results_dir))
    if args.run_id:
        results = {k: v for k, v in results.items() if k in set(args.run_id)}
        if not results:
            print(f"No matching run ids for period {args.period}", file=sys.stderr)
            return 2
    results = {k: v for k, v in results.items() if v.get("period") == args.period}
    if not args.run_id:
        results = _select_canonical(results, include_identified=args.include_identified)
    if not results:
        print(f"No results for period {args.period}", file=sys.stderr)
        return 2

    ledger = Ledger()
    trial_sharpes = ledger.sharpes()
    null = None
    null_path = Path(args.results_dir) / f"null_{args.period}.json"
    if null_path.exists():
        null = json.loads(null_path.read_text(encoding="utf-8"))
    power_path = Path(args.results_dir) / "power_analysis.json"
    power = json.loads(power_path.read_text(encoding="utf-8")) if power_path.exists() else None
    config = load_config("experiment.yaml")

    rows = []
    by_strategy: dict[str, dict] = {}
    for run_id, result in results.items():
        by_strategy.setdefault(result["strategy"], result)
    baseline = by_strategy.get("equal_weight_buy_hold")
    simple_candidates = [
        by_strategy[name]
        for name in ("momentum", "low_vol", "equal_weight_rebalanced")
        if name in by_strategy
    ]
    best_simple = (
        max(simple_candidates, key=lambda r: float(r["metrics"].get("sharpe") or -1e9))
        if simple_candidates
        else baseline
    )

    for run_id, result in sorted(results.items(), key=lambda kv: kv[1]["strategy"]):
        ret = _returns(result)
        metrics = result["metrics"]
        ci = bootstrap_ci(
            ret,
            n_boot=args.n_boot,
            block_length=args.block_length,
            seed=args.seed,
        )
        ann_ret_ci = _block_bootstrap_stat(
            ret,
            lambda r: float((np.prod(1.0 + r) ** (52.0 / max(1, len(r))) - 1.0)),
            n_boot=args.n_boot,
            block_length=args.block_length,
            seed=args.seed + 1,
        )
        vol_ci = _block_bootstrap_stat(
            ret,
            lambda r: float(np.std(r, ddof=1) * math.sqrt(52.0)) if len(r) > 1 else float("nan"),
            n_boot=args.n_boot,
            block_length=args.block_length,
            seed=args.seed + 2,
        )
        dd_ci = _block_bootstrap_stat(
            ret,
            _max_drawdown_from_returns,
            n_boot=args.n_boot,
            block_length=args.block_length,
            seed=args.seed + 3,
        )
        vs_eq = (
            paired_bootstrap_diff(
                ret,
                _returns(baseline),
                n_boot=args.n_boot,
                block_length=args.block_length,
                seed=args.seed,
            )
            if baseline is not None and result is not baseline
            else None
        )
        vs_simple = (
            paired_bootstrap_diff(
                ret,
                _returns(best_simple),
                n_boot=args.n_boot,
                block_length=args.block_length,
                seed=args.seed,
            )
            if best_simple is not None and result is not best_simple
            else None
        )
        percentile = (
            random_agent_percentile(float(metrics.get("sharpe", math.nan)), null.get("sharpe", []))
            if null
            else None
        )
        dsr = deflated_sharpe_ratio(
            float(metrics.get("sharpe", math.nan)),
            trial_sharpes,
            int(metrics.get("n_periods", 0)),
        )
        rows.append(
            {
                "run_id": run_id,
                "strategy": result["strategy"],
                "model": result.get("model"),
                "mode": result.get("mode"),
                "ann_return": metrics.get("annualized_return"),
                "ann_vol": metrics.get("annualized_volatility"),
                "sharpe": metrics.get("sharpe"),
                "sharpe_ci": [ci["low"], ci["high"]],
                "ann_return_ci": [ann_ret_ci["low"], ann_ret_ci["high"]],
                "ann_vol_ci": [vol_ci["low"], vol_ci["high"]],
                "max_dd": metrics.get("max_drawdown"),
                "max_dd_ci": [dd_ci["low"], dd_ci["high"]],
                "turnover": metrics.get("annualized_turnover"),
                "vs_equal_weight": vs_eq,
                "vs_best_simple": vs_simple,
                "null_percentile": percentile,
                "deflated_sharpe": dsr,
                "n_periods": metrics.get("n_periods"),
            }
        )

    rows.sort(key=lambda r: (-(r["sharpe"] if r["sharpe"] is not None else -1e9)))
    target_row = next((r for r in rows if r["strategy"] == "debate"), rows[0])
    target_ret = _returns(results[target_row["run_id"]])
    comparisons = []
    for row in rows:
        if row["run_id"] == target_row["run_id"]:
            continue
        other = results[row["run_id"]]
        comparisons.append(
            {
                "against_strategy": row["strategy"],
                "against_run_id": row["run_id"],
                "sharpe_diff": paired_bootstrap_diff(
                    target_ret,
                    _returns(other),
                    n_boot=args.n_boot,
                    block_length=args.block_length,
                    seed=args.seed,
                ),
            }
        )

    def _load_extra(name: str):
        path = Path(args.results_dir) / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    gap_by_period = {}
    for path in sorted(Path(args.results_dir).glob("contamination_gap*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("period"):
            gap_by_period[str(data["period"])] = data
    gap = (
        gap_by_period.get("precutoff_holdout")
        or gap_by_period.get("dev")
        or (next(iter(gap_by_period.values())) if gap_by_period else None)
    )
    extras = {
        "contamination_gap": gap,
        "contamination_gap_by_period": gap_by_period,
        "probes": _load_extra("probes.json"),
        "leakage_scan": _load_extra("leakage_scan.json"),
        "robustness": _load_extra("robustness.json"),
        "tuning": _load_extra("tuning_report.json"),
    }
    debate_row = next((r for r in rows if r["strategy"] == "debate"), None)
    best_simple_row = next(
        (r for r in rows if best_simple is not None and r["run_id"] == best_simple["run_id"]),
        None,
    )
    criteria = {}
    if debate_row and args.period == "clean_test":
        criteria["1_beats_baselines"] = bool(
            debate_row["vs_equal_weight"] is not None
            and debate_row["vs_equal_weight"]["low"] > 0
            and debate_row["vs_best_simple"] is not None
            and debate_row["vs_best_simple"]["low"] > 0
        )
        criteria["2_null_percentile"] = bool(
            debate_row["null_percentile"] is not None
            and debate_row["null_percentile"] >= 95.0
        )
        robustness_rows = {
            str(row.get("variant")): row
            for row in (extras["robustness"] or {}).get("rows", [])
        }
        criteria["3_robustness"] = bool(
            all(
                name in robustness_rows and robustness_rows[name].get("verdict_positive")
                for name in ("costs_2x", "rebalance_wednesday")
            )
        )
        gap = extras["contamination_gap"]
        thresholds = config.get("contamination_thresholds") or {}
        probe_periods = (extras["probes"] or {}).get("by_period", {})
        probe_values = (
            [probe_periods["clean_test"]]
            if "clean_test" in probe_periods
            else ([probe_periods["dev"]] if "dev" in probe_periods else list(probe_periods.values()))
        )
        max_probe_rate = max(
            (float(v.get("asset_identification_rate", 0.0)) for v in probe_values),
            default=0.0,
        )
        scan_runs = (extras["leakage_scan"] or {}).get("runs", [])
        clean_scan = [r for r in scan_runs if r.get("period") == "clean_test"]
        if not clean_scan:
            clean_scan = scan_runs
        scan_flags = sum(int(r.get("identifier_flags", 0)) + int(r.get("unmatched_numeric_levels", 0)) for r in clean_scan)
        clean_evidence_complete = bool(
            args.period != "clean_test"
            or (
                "clean_test" in probe_periods
                and any(r.get("period") == "clean_test" for r in scan_runs)
                and "precutoff_holdout" in gap_by_period
            )
        )
        probe_limit = float(thresholds.get("max_clean_probe_identification_rate", 0.15))
        flag_limit = int(thresholds.get("max_unexplained_clean_transcript_flags", 0))
        criteria["4_contamination"] = bool(
            clean_evidence_complete
            and gap
            and gap.get("passes_threshold")
            and extras["probes"] is not None
            and extras["leakage_scan"] is not None
            and max_probe_rate <= probe_limit
            and scan_flags <= flag_limit
        )
        criteria["5_deflated_sharpe"] = bool(debate_row["deflated_sharpe"]["passes"])
    payload = {
        "period": args.period,
        "n_results": len(rows),
        "ledger_trials": len(trial_sharpes),
        "null_runs": len(null.get("sharpe", [])) if null else 0,
        "power": power,
        "rows": rows,
        "best_simple_run_id": best_simple["run_id"] if best_simple else None,
        "target_run_id": target_row["run_id"],
        "comparisons_vs_target": comparisons,
        "criteria": criteria,
        "criteria_all_pass": bool(criteria) and all(criteria.values()),
        "claims_allowed": bool(
            args.period == "clean_test"
            and not load_period_table().provisional
            and criteria
            and all(criteria.values())
        ),
        "extras": extras,
        "caveats": [
            "Bootstrap intervals assume weekly blocks are approximately stationary.",
            "Dev/holdout results are not clean-test evidence.",
            "DSR uses the entire trial ledger, including baselines and null.",
            "Provisional periods invalidate final claims.",
        ],
    }
    out = Path(args.out) if args.out else Path(args.results_dir) / f"analysis_{args.period}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        f"# Analysis: {args.period}",
        "",
        f"- Results: {len(rows)}",
        f"- Ledger trials: {len(trial_sharpes)}",
        f"- Null runs: {payload['null_runs']}",
        "",
        "| strategy | Sharpe | 95% CI | AnnRet | MaxDD | vs EQ p | Null pct | DSR pass |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        vs_eq = r["vs_equal_weight"] or {}
        ci_low, ci_high = r["sharpe_ci"]
        lines.append(
            f"| {r['strategy']} | {_fmt(r['sharpe'])} | "
            f"[{_fmt(ci_low)}, {_fmt(ci_high)}] | {_fmt(r['ann_return'])} | "
            f"{_fmt(r['max_dd'])} | {_fmt(vs_eq.get('p_value_two_sided'))} | "
            f"{_fmt(r['null_percentile'])} | {r['deflated_sharpe']['passes']} |"
        )
    if power:
        lines += [
            "",
            f"Minimum detectable Sharpe gap: {_fmt(power.get('mde', {}).get('mde_sharpe'))} "
            f"over {power.get('n_weeks')} weeks.",
        ]
    md = Path(args.results_dir) / f"analysis_{args.period}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
