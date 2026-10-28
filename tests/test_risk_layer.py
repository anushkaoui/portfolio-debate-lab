from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given, settings, strategies as st

from core.risk_layer import (
    RiskConfig,
    RiskInputError,
    RiskInputs,
    RiskLayer,
)


TICKERS = ["T0", "T1", "T2", "T3", "T4", "T5"]
GROUPS = {t: ("A" if i < 3 else "B") for i, t in enumerate(TICKERS)}


@pytest.fixture(scope="module")
def layer():
    return RiskLayer(RiskConfig(), TICKERS, GROUPS)


def _inputs(target, current, *, vols=None, adv=None, drawdown=0.0, days=None):
    target_sum = sum(target.values())
    assert target_sum <= 1.0 + 1e-9
    current_sum = sum(current.values())
    assert current_sum <= 1.0 + 1e-9
    return RiskInputs(
        as_of=date(2023, 1, 6),
        target_weights=target,
        current_weights=current,
        portfolio_value=1_000_000.0,
        prices={t: 10.0 for t in TICKERS},
        volatilities=vols or {},
        adv_notional=adv or {},
        drawdown=drawdown,
        days_since_breaker=days,
    )


def test_caps_single_name_group_gross(layer):
    inputs = _inputs(
        {t: 0.15 for t in TICKERS},
        {t: 0.0 for t in TICKERS},
    )
    verdict = layer.apply(inputs)
    assert not layer.breaches(verdict, inputs)
    assert max(verdict.approved_weights.values()) <= 0.10 + 1e-9
    assert sum(verdict.approved_weights.values()) <= 1.0 + 1e-9
    assert verdict.gross_exposure <= 1.0 + 1e-9


def test_vol_scaled_cap_and_stale_hold(layer):
    inputs = _inputs(
        {"T0": 0.2, "T1": 0.1},
        {"T0": 0.0, "T1": 0.08},
        vols={"T0": 1.0},
    )
    verdict = layer.apply(inputs)
    assert verdict.approved_weights["T0"] <= 0.05 + 1e-9
    stale = RiskInputs(**{**inputs.__dict__, "stale_tickers": frozenset({"T1"})})
    verdict_stale = layer.apply(stale)
    assert verdict_stale.approved_weights["T1"] == pytest.approx(0.08)


def test_drawdown_breaker_reduces_gross(layer):
    inputs = _inputs(
        {t: 0.08 for t in TICKERS},
        {t: 0.0 for t in TICKERS},
        drawdown=-0.20,
    )
    verdict = layer.apply(inputs)
    assert verdict.gross_exposure <= 0.60 + 1e-9
    assert not layer.breaches(verdict, inputs)
    halted = layer.apply(
        _inputs({t: 0.08 for t in TICKERS}, {t: 0.0 for t in TICKERS}, drawdown=-0.30)
    )
    assert halted.halted is True
    assert halted.gross_exposure <= 0.30 + 1e-9


def test_liquidity_limits_buys_not_risk_reducing_sells(layer):
    inputs = _inputs(
        {"T0": 0.10},
        {"T0": 0.0},
        adv={"T0": 1000.0},
    )
    verdict = layer.apply(inputs)
    assert verdict.approved_weights["T0"] <= 0.0001 + 1e-9
    sell_inputs = _inputs(
        {"T0": 0.0},
        {"T0": 0.10},
        adv={"T0": 1000.0},
    )
    sell_verdict = layer.apply(sell_inputs)
    assert sell_verdict.approved_weights["T0"] == pytest.approx(0.0, abs=1e-9)


def test_rejects_prose_nan_and_nonfinite(layer):
    with pytest.raises(RiskInputError):
        RiskInputs(
            as_of=date(2023, 1, 6),
            target_weights={"T0": "buy aggressively"},
            current_weights={t: 0.0 for t in TICKERS},
            portfolio_value=100.0,
            prices={t: 1.0 for t in TICKERS},
            volatilities={},
            adv_notional={},
        )
    with pytest.raises(RiskInputError):
        RiskInputs(
            as_of=date(2023, 1, 6),
            target_weights={"T0": float("nan")},
            current_weights={t: 0.0 for t in TICKERS},
            portfolio_value=100.0,
            prices={t: 1.0 for t in TICKERS},
            volatilities={},
            adv_notional={},
        )
    with pytest.raises(RiskInputError):
        RiskInputs(
            as_of=date(2023, 1, 6),
            target_weights={t: 0.0 for t in TICKERS},
            current_weights={t: 0.0 for t in TICKERS},
            portfolio_value=float("inf"),
            prices={t: 1.0 for t in TICKERS},
            volatilities={},
            adv_notional={},
        )


@settings(max_examples=1500, deadline=None)
@given(
    target=st.lists(st.floats(0, 0.3), min_size=6, max_size=6),
    current=st.lists(st.floats(0, 0.3), min_size=6, max_size=6),
    vols=st.lists(st.floats(0, 1.5), min_size=6, max_size=6),
    adv=st.lists(st.floats(0, 1e9), min_size=6, max_size=6),
    drawdown=st.floats(-0.4, 0.0),
    days=st.one_of(st.none(), st.integers(0, 100)),
    portfolio_value=st.floats(1e3, 1e7),
)
def test_zero_breaches_and_idempotence(
    layer, target, current, vols, adv, drawdown, days, portfolio_value
):
    tw = dict(zip(TICKERS, target))
    total = sum(tw.values())
    if total > 1.0:
        tw = {t: v / total for t, v in tw.items()}
    cw = dict(zip(TICKERS, current))
    total = sum(cw.values())
    if total > 1.0:
        cw = {t: v / total for t, v in cw.items()}
    inputs = RiskInputs(
        as_of=date(2023, 1, 6),
        target_weights=tw,
        current_weights=cw,
        portfolio_value=portfolio_value,
        prices={t: 10.0 for t in TICKERS},
        volatilities=dict(zip(TICKERS, vols)),
        adv_notional=dict(zip(TICKERS, adv)),
        drawdown=drawdown,
        days_since_breaker=days,
    )
    verdict = layer.apply(inputs)
    assert layer.breaches(verdict, inputs) == []
    assert all(v >= -1e-12 for v in verdict.approved_weights.values())
    assert sum(verdict.approved_weights.values()) <= 1.0 + 1e-6

    again = layer.apply(
        RiskInputs(
            as_of=date(2023, 1, 6),
            target_weights=verdict.approved_weights,
            current_weights=verdict.approved_weights,
            portfolio_value=portfolio_value,
            prices={t: 10.0 for t in TICKERS},
            volatilities=dict(zip(TICKERS, vols)),
            adv_notional=dict(zip(TICKERS, adv)),
            drawdown=drawdown,
            days_since_breaker=days,
        )
    )
    assert layer.breaches(again, None) == []
    for t in TICKERS:
        assert again.approved_weights[t] == pytest.approx(
            verdict.approved_weights[t], abs=1e-6
        )
