#!/usr/bin/env python3
"""Contamination probes: can the model identify assets/periods or predict moves?"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.llm import make_client  # noqa: E402
from agents.prompts import build_prompt  # noqa: E402
from agents.schemas import IdentificationGuess, ProbePrediction  # noqa: E402
from core.anonymize import build_anonymizer_for_universe  # noqa: E402
from core.config import ROOT as CORE_ROOT, config_hash, load_config, load_universe, stable_json  # noqa: E402
from core.context import ContextBuilder  # noqa: E402
from core.freeze import verify_freeze_lock  # noqa: E402
from core.ledger import Ledger  # noqa: E402
from core.periods import decision_dates_for_period, load_period_table  # noqa: E402
from core.timegate import TimeGate, load_price_file  # noqa: E402


def _probe_context(builder: ContextBuilder, ticker: str, as_of: date, snapshot) -> tuple[dict, str]:
    packet = builder.build(
        as_of,
        portfolio_weights={},
        cash_weight=1.0,
        snapshot=snapshot,
    )
    code = packet.ticker_to_code[ticker]
    context = {
        "schema_version": "probe-1.0",
        "time": {"as_of": "t-0", "step": "1w"},
        "universe": [code],
        "features": {code: packet.context["features"][code]},
        "instructions": "Identify the asset if you can, state unknown if not.",
    }
    # The as_of is deliberately excluded from the visible context but included
    # in the cache key, matching the time gate's cache key discipline.
    context_hash = hashlib.sha256(
        stable_json({"context": context, "as_of": as_of.isoformat()}).encode("utf-8")
    ).hexdigest()
    return context, context_hash


def _next_move_sign(execution, ticker: str, as_of: date, horizon: int = 5) -> str | None:
    start = execution.close(ticker, as_of) or execution.last_close(ticker, as_of)
    if start is None:
        return None
    day = as_of
    try:
        for _ in range(horizon):
            day = execution.next_trading_day(day)
    except Exception:
        return None
    end = execution.close(ticker, day)
    if end is None:
        return None
    move = end / start - 1.0
    if move > 0.01:
        return "up"
    if move < -0.01:
        return "down"
    return "flat"


def _normalise(text: str | None) -> str:
    return (text or "").strip().lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", default="stress_holdout,dev,precutoff_holdout")
    parser.add_argument("--data", default=None)
    parser.add_argument("--n", type=int, default=20, help="probe dates per period")
    parser.add_argument("--mode", default="mock", choices=["mock", "live"])
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--final", action="store_true", help="allow clean_test/forward probes after an approved freeze")
    parser.add_argument("--freeze-path", default=None)
    args = parser.parse_args(argv)

    requested = [p.strip() for p in args.periods.split(",") if p.strip()]
    if any(p in {"clean_test", "forward"} for p in requested):
        if not args.final:
            print("FAIL: clean_test/forward probes require --final.", file=sys.stderr)
            return 2
        ok, reason = verify_freeze_lock(args.freeze_path, require_approved=True)
        if not ok:
            print(f"FAIL: {reason}", file=sys.stderr)
            return 2

    table = load_period_table()
    universe = load_universe()
    risk_cfg = load_config("risk_limits.yaml")
    anonymizer = build_anonymizer_for_universe(universe)
    client = make_client(args.mode, max_calls=args.max_calls)
    prices = load_price_file(args.data)
    records = []
    rng = np.random.default_rng(args.seed)
    code_to_ticker = anonymizer.code_to_ticker()
    company_by_ticker = {t: universe["companies"].get(t, t) for t in universe["tickers"]}

    for period_name in requested:
        period = table.get(period_name)
        frame = prices if period.end is None else prices[prices["date"] <= period.end]
        gate = TimeGate(frame, universe=universe["tickers"], max_visible_date=period.end)
        execution = gate.execution_prices()
        builder = ContextBuilder(gate, anonymizer, universe, risk_cfg)
        dates = decision_dates_for_period(table, period_name)
        if not dates:
            continue
        selected = [dates[i] for i in sorted(rng.choice(len(dates), size=min(args.n, len(dates)), replace=False))]
        for as_of in selected:
            ticker = str(rng.choice(universe["tickers"]))
            snapshot = gate.get(as_of).snapshot(universe["tickers"], stale_days=int(risk_cfg.get("stale_data_days", 10)))
            context, context_hash = _probe_context(builder, ticker, as_of, snapshot)
            prompt = build_prompt(
                "identification_guess",
                schema_name="IdentificationGuess",
                context=context,
                schema_json=stable_json(IdentificationGuess.model_json_schema()),
            )
            ident = client.call(
                role="identification_guess",
                system=prompt.system,
                user=prompt.user,
                schema=IdentificationGuess,
                params={"temperature": 0.0, "max_tokens": 300},
                as_of=as_of,
                context_hash=context_hash,
                payload=prompt.payload,
            ).value
            prompt2 = build_prompt(
                "probe_prediction",
                schema_name="ProbePrediction",
                context=context,
                schema_json=stable_json(ProbePrediction.model_json_schema()),
            )
            pred = client.call(
                role="probe_prediction",
                system=prompt2.system,
                user=prompt2.user,
                schema=ProbePrediction,
                params={"temperature": 0.0, "max_tokens": 300},
                as_of=as_of,
                context_hash=context_hash,
                payload=prompt2.payload,
            ).value
            guess = _normalise(ident.guessed_asset)
            ticker_l = ticker.lower()
            name_l = company_by_ticker[ticker].lower()
            correct_asset = bool(
                ident.recognizes_asset
                and (
                    guess == ticker_l
                    or ticker_l in guess
                    or (name_l and name_l in guess)
                )
            )
            period_guess = _normalise(ident.guessed_period)
            guessed_year = re.search(r"(20\d{2})", period_guess)
            actual_year = str(as_of.year)
            correct_period = bool(guessed_year and guessed_year.group(1) == actual_year)
            actual_move = _next_move_sign(execution, ticker, as_of)
            correct_move = actual_move is not None and pred.next_move == actual_move
            records.append(
                {
                    "period": period_name,
                    "as_of": as_of.isoformat(),
                    "ticker": ticker,
                    "code": anonymizer.code(ticker),
                    "recognizes_asset": ident.recognizes_asset,
                    "correct_asset": correct_asset,
                    "guessed_asset": ident.guessed_asset,
                    "guessed_period": ident.guessed_period,
                    "period_attempted": period_guess not in {"", "unknown", "none"},
                    "correct_period": correct_period,
                    "confidence": ident.confidence,
                    "reasoning": ident.reasoning,
                    "prediction": pred.next_move,
                    "actual_move": actual_move,
                    "correct_move": correct_move,
                    "prediction_confidence": pred.confidence,
                }
            )

    by_period = {}
    for period_name in sorted({r["period"] for r in records}):
        subset = [r for r in records if r["period"] == period_name]
        by_period[period_name] = {
            "n": len(subset),
            "asset_identification_rate": float(np.mean([r["correct_asset"] for r in subset])),
            "asset_attempt_rate": float(np.mean([r["recognizes_asset"] for r in subset])),
            "period_attempt_rate": float(np.mean([r["period_attempted"] for r in subset])),
            "period_year_accuracy": float(np.mean([r["correct_period"] for r in subset])),
            "move_accuracy": float(np.mean([r["correct_move"] for r in subset])),
            "chance_asset_rate": 1.0 / max(1, len(universe["tickers"])),
            "chance_move_rate": 1.0 / 3.0,
        }
    payload = {
        "mode": args.mode,
        "model": client.provider.model,
        "provider": client.provider.name,
        "chance_asset_rate": 1.0 / max(1, len(universe["tickers"])),
        "by_period": by_period,
        "records": records,
        "calls": client.n_calls,
        "provider_calls": client.n_provider_calls,
        "cost_usd": client.total_cost,
        "provisional_periods": table.provisional,
    }
    out = Path(args.out) if args.out else CORE_ROOT / "data" / "results" / "probes.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if not args.no_ledger:
        Ledger().append(
            strategy="memorization_probe",
            period=",".join(by_period),
            model=client.provider.model,
            mode=args.mode,
            config_hash=config_hash(),
            metrics=by_period,
            cost_usd=client.total_cost,
            calls=client.n_calls,
            provider_calls=client.n_provider_calls,
            notes="Memorization probes; builder never runs these on clean/sealed data.",
        )
    print(json.dumps(by_period, indent=2, sort_keys=True))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
