from __future__ import annotations

from datetime import date

import pytest

from core.simulator import PortfolioSimulator, SimulationError
from core.timegate import ExecutionPrices


def _sim(prices, zero_costs, capital=100_000.0):
    return PortfolioSimulator(
        ExecutionPrices(prices),
        ["AAA", "BBB"],
        zero_costs,
        initial_capital=capital,
        max_order_fraction_of_adv=1.0,
    )


def test_fills_next_open_and_accounting(prices, zero_costs):
    sim = _sim(prices, zero_costs)
    dates = sorted(prices["date"].unique())
    decision = dates[0]
    fill = dates[1]
    reb = sim.rebalance({"AAA": 0.5, "BBB": 0.5}, decision)
    assert reb.state.as_of == fill
    assert len(reb.fills) == 2
    assert all(f.fill_date == fill for f in reb.fills)
    assert all(f.decision_date == decision for f in reb.fills)
    assert sum(reb.state.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert reb.state.cash == pytest.approx(0.0, abs=1e-6)
    position_value = sum(reb.state.positions[t] * reb.state.prices[t] for t in ("AAA", "BBB"))
    assert reb.state.equity == pytest.approx(reb.state.cash + position_value, abs=1e-6)
    assert reb.costs == 0.0


def test_sells_before_buys_and_turnover(prices, zero_costs):
    sim = _sim(prices, zero_costs)
    dates = sorted(prices["date"].unique())
    sim.rebalance({"AAA": 1.0}, dates[0])
    reb = sim.rebalance({"BBB": 1.0}, dates[2])
    assert not reb.skipped
    assert reb.state.cash >= 0.0
    assert reb.state.weights["AAA"] == pytest.approx(0.0, abs=1e-9)
    assert reb.state.weights["BBB"] > 0.99
    assert 0.0 < reb.turnover <= 1.5
    assert not reb.fills or reb.fills[0].side == "sell"


def test_costs_are_charged(prices):
    from core.simulator import CostModel

    sim = PortfolioSimulator(
        ExecutionPrices(prices),
        ["AAA"],
        CostModel(commission_bps=10.0, slippage_bps=10.0, cost_multiplier=1.0),
        initial_capital=100_000.0,
        max_order_fraction_of_adv=1.0,
    )
    reb = sim.rebalance({"AAA": 0.5}, sorted(prices["date"].unique())[0])
    assert reb.costs > 0
    assert reb.state.cumulative_costs == pytest.approx(reb.costs)
    assert reb.state.equity < 100_000.0


def test_rejects_leverage_short_and_bad_fill_time(prices, zero_costs):
    sim = _sim(prices, zero_costs)
    dates = sorted(prices["date"].unique())
    with pytest.raises(SimulationError):
        sim.rebalance({"AAA": 0.7, "BBB": 0.7}, dates[0])
    with pytest.raises(SimulationError):
        sim.rebalance({"AAA": -0.1}, dates[0])
    with pytest.raises(SimulationError):
        sim.rebalance({"AAA": 0.1}, dates[0], fill_date=dates[0])


def test_missing_ticker_data_is_skipped(prices, zero_costs):
    dates = sorted(prices["date"].unique())
    frame = prices.copy()
    frame.loc[(frame["date"] == dates[1]) & (frame["ticker"] == "BBB"), "open"] = float("nan")
    sim = _sim(frame, zero_costs)
    reb = sim.rebalance({"AAA": 0.5, "BBB": 0.5}, dates[0])
    assert "BBB" in reb.skipped
    assert reb.state.weights.get("BBB", 0.0) == 0.0
