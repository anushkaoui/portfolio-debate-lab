"""End-to-end backtest runner for baselines, ablations and the debate system."""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from agents.debate import DebateEngine
from agents.llm import CachedLLM, make_client
from baselines.base import StrategyContext
from baselines.equal_weight import EqualWeightBuyHold, EqualWeightRebalanced
from baselines.low_vol import LowVolatility
from baselines.momentum import Momentum
from baselines.random_agent import RandomAgent
from baselines.single_llm import ensemble_weights, single_llm_weights
from core.anonymize import build_anonymizer_for_universe
from core.config import (
    ROOT,
    config_hash,
    git_commit,
    load_config,
    load_universe,
)
from core.context import ContextBuilder, ContextPacket
from core.ledger import Ledger
from core.metrics import conviction_calibration, summarize
from core.period_guard import PeriodGuard
from core.periods import (
    PeriodTable,
    decision_dates_for_period,
    load_period_table,
    weekly_decision_dates,
)
from core.risk_layer import RiskInputs, RiskLayer
from core.simulator import CostModel, PortfolioSimulator
from core.timegate import ExecutionPrices, TimeGate, load_price_file
from core.trace import TraceStore


class BacktestError(RuntimeError):
    pass


LLM_STRATEGIES = {
    "debate",
    "debate_no_risk_manager",
    "debate_one_analyst",
    "debate_no_debate",
    "single_llm",
    "ensemble",
}


@dataclass
class BacktestSpec:
    strategy: str
    period: str = "dev"
    mode: str = "mock"
    data_path: str | None = None
    max_decisions: int | None = None
    seed: int = 20261028
    identified: bool = False
    cost_multiplier: float | None = None
    rebalance_weekday: int = 4
    frequency: str = "weekly"
    universe_fraction: float = 1.0
    final: bool = False
    freeze_path: str | None = None
    run_id: str | None = None
    write_ledger: bool = True
    write_trace: bool = True
    write_results: bool = True
    max_llm_calls: int = 1200
    mock_seed: int = 0
    notes: str | None = None
    risk_free_annual: float = 0.0
    results_dir: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    tunable_overrides: dict[str, Any] | None = None


@dataclass
class BacktestResult:
    run_id: str
    strategy: str
    period: str
    metrics: dict
    equity: list[dict]
    config_hash: str
    model: str
    mode: str
    calls: int
    provider_calls: int
    cost_usd: float
    turnover: list[float]
    costs: list[float]
    warnings: list[str] = field(default_factory=list)
    ledger_trial: int | None = None
    decisions: int = 0
    flags: dict[str, int] = field(default_factory=dict)
    spec: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _subset_universe(universe: dict, fraction: float, seed: int) -> list[str]:
    tickers = list(universe["tickers"])
    if fraction >= 0.999:
        return tickers
    n = max(2, int(round(len(tickers) * fraction)))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(tickers, n))
    return chosen


def _strategy_object(
    name: str,
    *,
    tunables: Mapping[str, Any],
    seed: int,
):
    if name == "equal_weight_buy_hold":
        return EqualWeightBuyHold()
    if name == "equal_weight_rebalanced":
        return EqualWeightRebalanced()
    if name == "momentum":
        return Momentum(
            int(tunables.get("momentum_lookback_days", 252)),
            int(tunables.get("momentum_skip_days", 21)),
            int(tunables.get("top_k", 5)),
        )
    if name == "low_vol":
        return LowVolatility(
            int(tunables.get("low_vol_lookback_days", 63)),
            int(tunables.get("top_k", 5)),
        )
    if name == "random_agent":
        return RandomAgent(seed=seed)
    if name in LLM_STRATEGIES:
        return None
    raise BacktestError(f"Unknown strategy {name!r}")


def run_backtest(spec: BacktestSpec) -> BacktestResult:
    if spec.strategy in LLM_STRATEGIES and spec.mode not in {"mock", "live"}:
        raise BacktestError("mode must be mock or live")
    universe = load_universe()
    tunables = dict(load_config("tunables.yaml"))
    if spec.tunable_overrides:
        tunables.update(spec.tunable_overrides)
    risk_cfg = load_config("risk_limits.yaml")
    costs_cfg = dict(load_config("costs.yaml"))
    if spec.cost_multiplier is not None:
        costs_cfg["cost_multiplier"] = float(spec.cost_multiplier)
    table: PeriodTable = load_period_table()
    guard = PeriodGuard(
        table,
        spec.period,
        final=spec.final,
        freeze_path=spec.freeze_path,
    )

    tickers = _subset_universe(universe, spec.universe_fraction, spec.seed)
    if spec.universe_fraction < 0.999:
        universe = dict(universe)
        universe["tickers"] = tickers
        universe["group_map"] = {t: universe["group_map"][t] for t in tickers}
        universe["companies"] = {t: universe["companies"][t] for t in tickers}
        universe["groups"] = {
            g: [t for t in members if t in tickers]
            for g, members in universe["groups"].items()
            if any(t in tickers for t in members)
        }

    period = table.get(spec.period)
    data_path = Path(spec.data_path) if spec.data_path else None
    prices = load_price_file(data_path)
    if period.end is not None:
        prices = prices[prices["date"] <= period.end]
    prices = prices[prices["ticker"].isin(tickers)]
    if prices.empty:
        raise BacktestError(f"No prices for period {spec.period} and chosen universe")

    gate = TimeGate(prices, universe=tickers, max_visible_date=period.end)
    execution = gate.execution_prices()
    cost_model = CostModel.from_config(costs_cfg)
    simulator = PortfolioSimulator(
        execution,
        tickers,
        cost_model,
        initial_capital=float(load_config("experiment.yaml").get("initial_capital", 100000)),
        max_order_fraction_of_adv=float(risk_cfg.get("max_order_fraction_of_adv", 0.10)),
    )
    risk = RiskLayer.from_config(risk_cfg, tickers, universe["group_map"])
    anonymizer = build_anonymizer_for_universe(load_universe())
    context_builder = ContextBuilder(gate, anonymizer, universe, risk_cfg)
    stale_days = int(risk_cfg.get("stale_data_days", 10))

    dates = decision_dates_for_period(
        table, spec.period, spec.frequency, spec.rebalance_weekday
    )
    if spec.start_date:
        start = date.fromisoformat(spec.start_date)
        dates = [d for d in dates if d >= start]
    if spec.end_date:
        end = date.fromisoformat(spec.end_date)
        dates = [d for d in dates if d <= end]
    if not dates:
        raise BacktestError(f"No decision dates in period {spec.period}")
    execute_dates = dates[:-1]
    if spec.max_decisions is not None:
        execute_dates = execute_dates[: int(spec.max_decisions)]
        dates = dates[: len(execute_dates) + 1]
    # A final valuation date may be needed when max_decisions is used.
    if not execute_dates:
        execute_dates = []
        dates = dates[:1]

    run_id = spec.run_id or (
        f"{spec.strategy}__{spec.period}__{spec.mode}__"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}__{uuid.uuid4().hex[:6]}"
    )
    strategy_obj = _strategy_object(spec.strategy, tunables=tunables, seed=spec.seed)
    client: CachedLLM | None = None
    debate_engine: DebateEngine | None = None
    if spec.strategy in LLM_STRATEGIES:
        client = make_client(
            spec.mode,
            max_calls=spec.max_llm_calls,
            mock_seed=spec.mock_seed,
        )
        if spec.strategy in {"debate", "debate_no_risk_manager", "debate_one_analyst", "debate_no_debate"}:
            styles = ("trend",) if spec.strategy in {"debate_one_analyst", "debate_no_debate"} else ("trend", "contrarian")
            debate_engine = DebateEngine(
                client,
                tunables,
                analyst_styles=styles,
                enable_risk_manager=spec.strategy != "debate_no_risk_manager" and spec.strategy != "debate_no_debate",
                enable_rebuttal=spec.strategy not in {"debate_no_debate"},
                max_calls_per_decision=int(tunables.get("max_calls_per_decision", 8)),
            )

    trace = TraceStore() if spec.write_trace else None
    ledger = Ledger()
    if trace:
        trace.start_run(
            run_id=run_id,
            strategy=spec.strategy,
            period=spec.period,
            model=(client.provider.model if client else "code"),
            mode=spec.mode if client else "code",
            config_hash=config_hash(),
            git_commit=git_commit(),
            notes=spec.notes,
        )

    equity_points: list[tuple[date, float]] = []
    turnovers: list[float] = []
    costs: list[float] = []
    conviction_records: list[dict] = []
    flags: dict[str, int] = {}
    calls = 0
    provider_calls = 0
    cost_usd = 0.0
    high_water = simulator.initial_capital
    last_breaker_index: int | None = None
    warnings: list[str] = []
    decision_count = 0
    fill_count = 0

    def bump(flag_list) -> None:
        for f in flag_list or []:
            key = str(f).split(":", 1)[0]
            flags[key] = flags.get(key, 0) + 1

    for i, d in enumerate(dates):
        guard.validate(d)
        view = gate.get(d)
        snapshot = view.snapshot(tickers, stale_days=stale_days)
        state = simulator.mark_to_market(d)
        equity_points.append((d, state.equity))
        drawdown = state.equity / high_water - 1.0 if high_water > 0 else 0.0
        days_since_breaker: int | None = None
        if last_breaker_index is not None:
            days_since_breaker = i - last_breaker_index
        if (
            drawdown <= float((risk_cfg.get("drawdown_breaker") or {}).get("reduce_threshold", -0.15))
            and d in execute_dates
        ):
            if last_breaker_index is None:
                last_breaker_index = i
        high_water = max(high_water, state.equity)

        if d not in execute_dates:
            continue
        decision_count += 1
        current_weights = {t: float(state.weights.get(t, 0.0)) for t in tickers}
        cash_weight = max(0.0, 1.0 - sum(current_weights.values()))
        ctx = StrategyContext(
            as_of=d,
            view=view,
            snapshot=snapshot,
            current_weights=current_weights,
            portfolio_value=state.equity,
            drawdown=drawdown,
            days_since_breaker=days_since_breaker,
            universe=tuple(tickers),
        )

        packet: ContextPacket | None = None
        target_weights: dict[str, float] | None = None
        debate_payload: dict | None = None
        strategy_flags: list[str] = []
        call_delta = 0
        provider_delta = 0
        cost_delta = 0.0

        if spec.strategy in LLM_STRATEGIES:
            packet = context_builder.build(
                d,
                portfolio_weights=current_weights,
                cash_weight=cash_weight,
                drawdown=drawdown,
                days_since_breaker=days_since_breaker,
                portfolio_value=state.equity,
                identified=spec.identified,
                snapshot=snapshot,
            )
            assert client is not None
            start_calls = client.n_calls
            start_provider = client.n_provider_calls
            start_cost = client.total_cost
            if spec.strategy in {"single_llm", "ensemble"}:
                if spec.strategy == "single_llm":
                    res = single_llm_weights(
                        client,
                        packet=packet,
                        params=tunables,
                        as_of=d,
                        max_delta=float(tunables.get("max_analyst_delta", 0.08)),
                    )
                else:
                    res = ensemble_weights(
                        client,
                        packet=packet,
                        params=tunables,
                        as_of=d,
                        max_delta=float(tunables.get("max_analyst_delta", 0.08)),
                        n_samples=int(tunables.get("ensemble_samples", 3)),
                    )
                target_weights = res.target_weights
                strategy_flags = res.flags
                debate_payload = {"baseline": spec.strategy, "messages": res.messages, "flags": res.flags}
            else:
                assert debate_engine is not None
                debate = debate_engine.decide(packet, as_of=d)
                target_weights = debate.target_weights_real
                strategy_flags = debate.flags
                debate_payload = debate.to_dict()
                if d in execution.dates_set:
                    idx = execution.dates.index(d)
                    future = execution.dates[min(idx + 5, len(execution.dates) - 1)]
                    for proposal in getattr(debate, "proposals", []) or []:
                        ticker = packet.code_to_ticker.get(proposal.asset)
                        if ticker is None:
                            continue
                        p0 = execution.close(ticker, d)
                        p1 = execution.close(ticker, future)
                        delta = float(proposal.target_weight_delta)
                        if p0 and p1 and proposal.direction != "hold" and delta != 0.0:
                            conviction_records.append(
                                {
                                    "conviction": float(proposal.conviction),
                                    "realized_return": (
                                        float(p1 / p0 - 1.0) * (1.0 if delta > 0 else -1.0)
                                    ),
                                    "asset": ticker,
                                    "as_of": d.isoformat(),
                                }
                            )
            call_delta = client.n_calls - start_calls
            provider_delta = client.n_provider_calls - start_provider
            cost_delta = client.total_cost - start_cost
            calls += call_delta
            provider_calls += provider_delta
            cost_usd += cost_delta
        else:
            assert strategy_obj is not None
            decision = strategy_obj.propose(ctx)
            target_weights = decision.weights
            strategy_flags = list(decision.flags)
            debate_payload = {"baseline": spec.strategy, "reason": decision.reason, "flags": strategy_flags}

        bump(strategy_flags)
        if target_weights is None:
            turnovers.append(0.0)
            costs.append(0.0)
            if trace:
                trace.add_decision(
                    run_id=run_id,
                    decision_date=d.isoformat(),
                    fill_date=None,
                    context_hash=packet.context_hash if packet else None,
                    context=packet.context if packet else None,
                    debate=debate_payload,
                    risk=None,
                    fills=[],
                    portfolio=state.to_dict(),
                    flags=strategy_flags,
                    equity=state.equity,
                )
            continue

        risk_inputs = RiskInputs(
            as_of=d,
            target_weights=target_weights,
            current_weights=current_weights,
            portfolio_value=state.equity,
            prices=snapshot.closes,
            volatilities={
                t: float(v.get("vol_20d") or 0.0)
                for t, v in snapshot.features.items()
                if isinstance(v.get("vol_20d"), (int, float))
            },
            adv_notional=snapshot.dollar_volume,
            stale_tickers=frozenset(
                t for t, v in snapshot.features.items() if v.get("stale") or v.get("missing")
            ),
            drawdown=drawdown,
            days_since_breaker=days_since_breaker,
        )
        verdict = risk.apply(risk_inputs)
        problems = risk.breaches(verdict, risk_inputs)
        if problems:
            raise BacktestError("Risk layer produced a limit breach: " + "; ".join(problems[:5]))
        for intervention in verdict.interventions:
            flags[f"risk:{intervention.rule}"] = flags.get(f"risk:{intervention.rule}", 0) + 1
        fill_date = execution.next_trading_day(d)
        if period.end is not None and fill_date > period.end:
            warnings.append(f"skipped fill after period end on {fill_date.isoformat()}")
            fill_date = None
        if fill_date is None:
            turnovers.append(0.0)
            costs.append(0.0)
            continue
        reb = simulator.rebalance(verdict.approved_weights, d, fill_date=fill_date)
        fill_count += len(reb.fills)
        turnovers.append(reb.turnover)
        costs.append(reb.costs)
        if trace:
            trace.add_decision(
                run_id=run_id,
                decision_date=d.isoformat(),
                fill_date=fill_date.isoformat(),
                context_hash=packet.context_hash if packet else None,
                context=packet.context if packet else None,
                debate=debate_payload,
                risk=verdict.to_dict(),
                fills=reb.to_dict()["fills"],
                portfolio=reb.state.to_dict(),
                flags=strategy_flags,
                equity=state.equity,
            )
        if reb.skipped:
            bump([f"skipped_fill:{t}" for t in reb.skipped])
            warnings.append(f"{d.isoformat()}: skipped fills for {reb.skipped}")

    if not equity_points:
        raise BacktestError("Backtest produced no equity observations")
    equity_values = [v for _, v in equity_points]
    metrics = summarize(
        equity_values,
        turnovers=turnovers,
        costs=costs,
        initial_capital=simulator.initial_capital,
        risk_free_annual=spec.risk_free_annual,
    )
    metrics.update(
        {
            "strategy": spec.strategy,
            "period": spec.period,
            "mode": spec.mode,
            "model": client.provider.model if client else "code",
            "identified": bool(spec.identified),
            "final": bool(spec.final),
            "decisions": decision_count,
            "fills": fill_count,
            "calls": calls,
            "provider_calls": provider_calls,
            "conviction_calibration": conviction_calibration(conviction_records),
            "llm_cost_usd": round(cost_usd, 8),
            "total_simulated_costs": round(sum(costs), 6),
            "provisional_periods": table.provisional,
            "period_cutoff": table.cutoff.isoformat(),
            "period_underpowered": table.underpowered,
        }
    )
    result = BacktestResult(
        run_id=run_id,
        strategy=spec.strategy,
        period=spec.period,
        metrics=metrics,
        equity=[{"date": d.isoformat(), "equity": round(v, 6)} for d, v in equity_points],
        config_hash=config_hash(),
        model=metrics["model"],
        mode=spec.mode,
        calls=calls,
        provider_calls=provider_calls,
        cost_usd=cost_usd,
        turnover=turnovers,
        costs=costs,
        warnings=sorted(set(warnings)),
        decisions=decision_count,
        flags=flags,
        spec=asdict(spec),
    )

    if spec.write_ledger:
        entry = ledger.append(
            strategy=spec.strategy,
            period=spec.period,
            model=result.model,
            mode=spec.mode,
            config_hash=result.config_hash,
            metrics=metrics,
            cost_usd=cost_usd,
            calls=calls,
            provider_calls=provider_calls,
            run_id=run_id,
            notes=spec.notes,
            extra={
                "identified": spec.identified,
                "cost_multiplier": costs_cfg.get("cost_multiplier", 1.0),
                "rebalance_weekday": spec.rebalance_weekday,
                "universe_fraction": spec.universe_fraction,
                "frequency": spec.frequency,
                "max_decisions": spec.max_decisions,
            },
        )
        result.ledger_trial = int(entry["trial"])

    if trace:
        trace.finish_run(
            run_id,
            metrics=metrics,
            cost_usd=cost_usd,
            calls=calls,
            provider_calls=provider_calls,
            notes=spec.notes,
        )

    if spec.write_results:
        out_dir = Path(spec.results_dir) if spec.results_dir else ROOT / "data" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{run_id}.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    return result
