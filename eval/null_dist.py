#!/usr/bin/env python3
"""Random-agent null distribution through the same risk layer and simulator."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.backtest import BacktestSpec, run_backtest  # noqa: E402
from core.config import config_hash  # noqa: E402
from core.ledger import Ledger  # noqa: E402


def _run_one(job: dict) -> dict:
    spec = BacktestSpec(
        strategy="random_agent",
        period=job["period"],
        mode="mock",
        data_path=job["data"],
        max_decisions=job["max_decisions"],
        seed=job["seed"],
        final=job["final"],
        freeze_path=job["freeze_path"],
        write_ledger=False,
        write_trace=False,
        write_results=False,
    )
    res = run_backtest(spec)
    return {
        "seed": job["seed"],
        "sharpe": float(res.metrics.get("sharpe", float("nan"))),
        "total_return": float(res.metrics.get("total_return", float("nan"))),
        "max_drawdown": float(res.metrics.get("max_drawdown", float("nan"))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="dev")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--data", default=None)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--freeze-path", default=None)
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel worker processes (1 = sequential).",
    )
    args = parser.parse_args(argv)
    if args.mode == "live":
        print(
            "WARNING: live mode is ignored for the random null; no LLM is called.",
            file=sys.stderr,
        )

    jobs = [
        {
            "period": args.period,
            "data": args.data,
            "max_decisions": args.max_decisions,
            "seed": args.seed + i,
            "final": args.final,
            "freeze_path": args.freeze_path,
        }
        for i in range(args.runs)
    ]
    sharpes: list[float] = []
    totals: list[float] = []
    max_dds: list[float] = []
    results: list[dict] = []
    workers = max(1, min(int(args.workers), args.runs))
    if workers == 1:
        for i, job in enumerate(jobs):
            results.append(_run_one(job))
            if (i + 1) % max(1, args.runs // 20) == 0 or i + 1 == args.runs:
                print(f"{i + 1}/{args.runs} null runs complete", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_run_one, jobs, chunksize=max(1, args.runs // (workers * 4)))):
                results.append(row)
                if (i + 1) % max(1, args.runs // 20) == 0 or i + 1 == args.runs:
                    print(f"{i + 1}/{args.runs} null runs complete", flush=True)
    for row in results:
        sharpes.append(row["sharpe"])
        totals.append(row["total_return"])
        max_dds.append(row["max_drawdown"])

    arr = np.asarray(sharpes, dtype=float)
    arr = arr[np.isfinite(arr)]
    metrics = {
        "null_runs": int(len(arr)),
        "mean_sharpe": float(np.mean(arr)) if len(arr) else float("nan"),
        "median_sharpe": float(np.median(arr)) if len(arr) else float("nan"),
        "std_sharpe": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "p50_sharpe": float(np.quantile(arr, 0.50)) if len(arr) else float("nan"),
        "p95_sharpe": float(np.quantile(arr, 0.95)) if len(arr) else float("nan"),
        "p99_sharpe": float(np.quantile(arr, 0.99)) if len(arr) else float("nan"),
        "mean_total_return": float(np.nanmean(totals)),
        "median_max_drawdown": float(np.nanmedian(max_dds)),
        "period": args.period,
        "seed_start": args.seed,
    }
    payload = {
        "metrics": metrics,
        "sharpe": [float(x) for x in arr],
        "seeds": [args.seed + i for i in range(args.runs)],
        "provisional": args.period in {"clean_test", "forward"} and not args.final,
    }
    out = ROOT / "data" / "results" / f"null_{args.period}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if not args.no_ledger:
        Ledger().append(
            strategy="random_agent_null",
            period=args.period,
            model="code",
            mode="mock",
            config_hash=config_hash(),
            metrics=metrics,
            calls=0,
            provider_calls=0,
            notes=f"Null distribution with {args.runs} seeds; one ledger entry for the null config.",
            extra={"runs": args.runs, "seed_start": args.seed},
        )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
