from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from agents.llm import BudgetExceeded, CachedLLM, CallBudget, CacheStore, LLMError, make_client
from agents.mock_llm import MockProvider, mock_generate
from agents.schemas import Decision, Proposal
from agents.debate import DebateEngine
from core.config import load_config, load_universe
from core.periods import decision_dates_for_period, load_period_table
from core.timegate import TimeGate, load_price_file
from core.anonymize import build_anonymizer_for_universe
from core.context import ContextBuilder


def test_schema_rejects_leverage_and_nan():
    with pytest.raises(ValidationError):
        Decision(
            target_weights={"S01": 0.7, "S02": 0.7},
            rationale="x",
            dissent_noted=False,
            evidence=["x"],
            recognizes_asset=False,
        )
    with pytest.raises(ValidationError):
        Decision(
            target_weights={"S01": float("nan")},
            rationale="x",
            dissent_noted=False,
            evidence=["x"],
            recognizes_asset=False,
        )
    with pytest.raises(ValidationError):
        Proposal(
            agent="a",
            asset="S01",
            direction="increase",
            target_weight_delta=2.0,
            conviction=0.5,
            horizon_days=5,
            thesis="x",
            evidence=["x"],
            what_would_change_my_mind="x",
            recognizes_asset=False,
        )


def test_mock_trader_never_leverages():
    payload = {
        "context": {
            "universe": ["S01", "S02"],
            "portfolio": {"weights": {"S01": 0.9, "S02": 0.9}},
            "features": {
                "S01": {"position_cap": 0.5, "vol_20d": 0.2},
                "S02": {"position_cap": 0.5, "vol_20d": 0.2},
            },
            "risk_headroom": {"max_single_position": 0.5},
        },
        "proposals": [
            {"asset": "S01", "target_weight_delta": 0.5, "direction": "increase"},
            {"asset": "S02", "target_weight_delta": 0.5, "direction": "increase"},
        ],
        "critiques": [],
        "responses": [],
    }
    decision = mock_generate("trader", payload, {})
    weights = decision["target_weights"]
    assert sum(weights.values()) <= 1.0
    assert all(w >= 0 for w in weights.values())


def test_cache_and_budget(tmp_path):
    from agents.llm import Usage

    cache = CacheStore(tmp_path / "cache.sqlite")
    client = CachedLLM(MockProvider(), cache, CallBudget(10))
    schema = Decision
    payload = {
        "context": {"universe": ["S01"], "portfolio": {"weights": {}}},
        "proposals": [],
        "critiques": [],
        "responses": [],
    }
    kwargs = dict(
        role="trader",
        system="s",
        user="u",
        schema=schema,
        params={"temperature": 0.0},
        as_of=date(2023, 1, 6),
        context_hash="abc",
        payload=payload,
    )
    first = client.call(**kwargs)
    second = client.call(**kwargs)
    assert first.cached is False
    assert second.cached is True
    assert client.n_provider_calls == 1
    assert client.n_calls == 2

    budget_client = CachedLLM(MockProvider(), CacheStore(tmp_path / "b.sqlite"), CallBudget(1))
    budget_client.call(**kwargs)
    with pytest.raises(BudgetExceeded):
        budget_client.call(**{**kwargs, "context_hash": "different"})


def test_debate_is_bounded_and_returns_real_tickers():
    universe = load_universe()
    prices = load_price_file()
    table = load_period_table()
    as_of = decision_dates_for_period(table, "dev")[5]
    gate = TimeGate(prices, universe=universe["tickers"], max_visible_date=table.periods["dev"].end)
    anonymizer = build_anonymizer_for_universe(universe)
    builder = ContextBuilder(gate, anonymizer, universe, load_config("risk_limits.yaml"))
    packet = builder.build(as_of)
    client = make_client("mock", max_calls=20)
    params = load_config("tunables.yaml")
    engine = DebateEngine(client, params, max_calls_per_decision=5)
    result = engine.decide(packet, as_of=as_of)
    assert result.calls <= 5
    assert set(result.target_weights_real) <= set(universe["tickers"])
    assert sum(result.target_weights_real.values()) <= 1.0 + 1e-8
    assert result.messages
