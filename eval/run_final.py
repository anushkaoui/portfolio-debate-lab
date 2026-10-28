#!/usr/bin/env python3
"""Human-only final evaluation. Refuses to run without an approved freeze.

The builder must never execute this script. It is kept here so the human can
run it exactly once after review, and so every rerun is recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402
from core.config import ROOT as CORE_ROOT, load_config, load_universe  # noqa: E402
from core.freeze import load_freeze_lock, verify_freeze_lock  # noqa: E402
from core.ledger import Ledger  # noqa: E402
from core.metrics import bootstrap_ci, deflated_sharpe_ratio, equity_returns, random_agent_percentile  # noqa: E402
from core.periods import load_period_table  # noqa: E402
from eval import null_dist  # noqa: E402


def _loads(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-the-human", action="store_true", required=True)
    parser.add_argument("--strategy", default="debate")
    parser.add_argument("--period", default="clean_test", choices=["clean_test"])
    parser.add_argument("--data", required=True, help="sealed price file; never data/dev")
    parser.add_argument("--mode", default="live", choices=["mock", "live"])
    parser.add_argument("--freeze-path", default=None)
    parser.add_argument("--null-runs", type=int, default=1000)
    parser.add_argument("--skip-null", action="store_true")
    parser.add_argument("--declared-rerun", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--max-llm-calls", type=int, default=1200)
    parser.add_argument(
        "--allow-mock-final",
        action="store_true",
        help="explicitly allow a pipeline dry run; never research evidence",
    )
    args = parser.parse_args(argv)

    data = Path(args.data).resolve()
    if not data.exists():
        print(f"FAIL: sealed data not found at {data}", file=sys.stderr)
        return 2
    if "data/dev" in str(data) or "data\\dev" in str(data):
        print("FAIL: final evaluation requires sealed data, not data/dev.", file=sys.stderr)
        return 2

    ok, reason = verify_freeze_lock(args.freeze_path, require_approved=True)
    if not ok:
        print(f"FAIL: {reason}", file=sys.stderr)
        return 2
    lock = load_freeze_lock(args.freeze_path)
    table = load_period_table()
    if table.provisional:
        print("FAIL: period table is still provisional.", file=sys.stderr)
        return 2
    models = load_config("models.yaml").get("agent_model", {})
    if not models.get("verified"):
        print("FAIL: models.yaml cutoff is not verified.", file=sys.stderr)
        return 2
    if not models.get("pinned_snapshot"):
        print("FAIL: models.yaml must pin the exact served checkpoint.", file=sys.stderr)
        return 2
    if not models.get("cutoff_source_url") or not models.get("checked_on"):
        print("FAIL: models.yaml must record the cutoff source and checked-on date.", file=sys.stderr)
        return 2
    if lock.get("model", {}).get("documented_cutoff") != table.cutoff.isoformat():
        print("FAIL: freeze lock cutoff does not match periods.yaml.", file=sys.stderr)
        return 2
    dry_run = args.mode == "mock"
    if dry_run and not args.allow_mock_final:
        print(
            "FAIL: mock cannot support a final claim. Use --allow-mock-final only "
            "for an explicit pipeline dry run.",
            file=sys.stderr,
        )
        return 2
    if dry_run:
        print(
            "WARNING: mock final run is a pipeline dry run; it is never research evidence.",
            file=sys.stderr,
        )

    registry = CORE_ROOT / "data" / "final_run_registry.jsonl"
    registry.parent.mkdir(parents=True, exist_ok=True)
    if registry.exists():
        prior = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines() if line.strip()]
        if (
            any(p.get("period") == "clean_test" and not p.get("dry_run") for p in prior)
            and not args.declared_rerun
        ):
            print(
                "FAIL: a non-dry-run clean-test run is already recorded. Rerun only "
                "with --declared-rerun '<reason written before seeing results>'.",
                file=sys.stderr,
            )
            return 2
    run_id = f"{args.strategy}__clean_test__{args.mode}__final"
    spec = BacktestSpec(
        strategy=args.strategy,
        period="clean_test",
        mode=args.mode,
        data_path=str(data),
        final=True,
        freeze_path=args.freeze_path,
        run_id=run_id,
        write_ledger=True,
        write_trace=True,
        write_results=True,
        seed=args.seed,
        max_llm_calls=args.max_llm_calls,
        notes=f"FINAL human-run evaluation. declared_rerun={args.declared_rerun!r}",
    )
    result = run_backtest(spec)
    with registry.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "run_id": result.run_id,
                    "period": "clean_test",
                    "mode": args.mode,
                    "dry_run": dry_run,
                    "declared_rerun": args.declared_rerun,
                    "freeze_sha256": lock["combined_sha256"],
                    "data_path": str(data),
                },
                sort_keys=True,
            )
            + "\n"
        )

    null_error = None
    if not args.skip_null:
        try:
            null_dist.main(
                [
                    "--period",
                    "clean_test",
                    "--runs",
                    str(args.null_runs),
                    "--data",
                    str(data),
                    "--seed",
                    str(args.seed),
                    "--final",
                    "--freeze-path",
                    str(args.freeze_path) if args.freeze_path else str(CORE_ROOT / "freeze.lock"),
                ]
            )
        except Exception as exc:  # keep the final report even if the null worker pool fails
            null_error = f"{type(exc).__name__}: {exc}"
            print(f"WARNING: null distribution failed: {null_error}", file=sys.stderr)

    returns = equity_returns([p["equity"] for p in result.equity]).to_numpy()
    ci = bootstrap_ci(returns, n_boot=5000)
    null_path = CORE_ROOT / "data" / "results" / "null_clean_test.json"
    null = _loads(null_path) if null_path.exists() else None
    percentile = (
        random_agent_percentile(float(result.metrics.get("sharpe", float("nan"))), null.get("sharpe", []))
        if null
        else float("nan")
    )
    dsr = deflated_sharpe_ratio(
        float(result.metrics.get("sharpe", float("nan"))),
        Ledger().sharpes(),
        int(result.metrics.get("n_periods", 0)),
    )
    power_path = CORE_ROOT / "data" / "results" / "power_analysis.json"
    power = _loads(power_path) if power_path.exists() else None
    report = {
        "run_id": result.run_id,
        "metrics": result.metrics,
        "sharpe_ci": ci,
        "null_percentile": percentile,
        "deflated_sharpe": dsr,
        "power": power,
        "freeze_sha256": lock["combined_sha256"],
        "warnings": result.warnings,
        "null_error": null_error,
        "dry_run": dry_run,
        "note": (
            "Auto-generated after the single clean-test run. The human must finish "
            "the interpretation after checking novel post-cutoff sources and must "
            "not rerun to improve results."
        ),
    }
    suffix = "_dry_run" if dry_run else ""
    out = CORE_ROOT / "data" / "results" / f"final_auto_report{suffix}.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        "# Final evaluation (auto-generated)",
        "",
        f"- Run: {result.run_id}",
        f"- Sharpe: {result.metrics.get('sharpe'):.3f} "
        f"(95% block-bootstrap CI {ci['low']:.3f} to {ci['high']:.3f})",
        f"- Annualized return: {result.metrics.get('annualized_return'):.3%}",
        f"- Max drawdown: {result.metrics.get('max_drawdown'):.3%}",
        f"- Random-null percentile: {percentile:.1f}",
        f"- Deflated Sharpe probability: {dsr['deflated_sharpe_probability']:.3f}",
        f"- Power MDE: {((power or {}).get('mde') or {}).get('mde_sharpe')}",
        f"- Trials in ledger: {dsr['n_trials']}",
        "",
        "## Warnings",
        "",
        *[f"- {w}" for w in result.warnings],
        "",
        "## Human interpretation (fill in)",
        "",
        "1. What the result does and does not establish:",
        "2. Post-cutoff novelty check for top contributors:",
        "3. Feasibility and zero-cost audit:",
        "4. Alpha decay after 2024-12-31 freeze:",
    ]
    (CORE_ROOT / "data" / "results" / f"final_auto_report{suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
