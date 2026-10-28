from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.timegate import DataError, LookAheadError, TimeGate


def test_no_future_rows_and_sabotage(prices):
    gate = TimeGate(prices, universe=["AAA", "BBB"])
    as_of = sorted(prices["date"].unique())[20]
    view = gate.get(as_of)
    assert max(view.rows()["date"]) <= as_of
    with pytest.raises(LookAheadError):
        view._slice(as_of + timedelta(days=1))
    snap = view.snapshot(["AAA", "BBB"])
    assert snap.as_of == as_of
    assert set(snap.features) == {"AAA", "BBB"}
    assert "ret_1w" in snap.features["AAA"]
    assert snap.features["AAA"]["last_observation"] <= as_of


def test_features_are_finite_and_relative(prices):
    gate = TimeGate(prices, universe=["AAA", "BBB"])
    snap = gate.get(sorted(prices["date"].unique())[-1]).snapshot(["AAA", "BBB"])
    for ticker in ("AAA", "BBB"):
        f = snap.features[ticker]
        assert f["last_close"] > 0
        assert len(f["price_index_26w"]) > 0
        assert f["rank_ret_1m"] in (0.0, 1.0)
    assert snap.closes["AAA"] > 0


def test_execution_next_day_and_last_close(prices):
    gate = TimeGate(prices, universe=["AAA", "BBB"])
    execution = gate.execution_prices()
    dates = sorted(prices["date"].unique())
    assert execution.next_trading_day(dates[0]) == dates[1]
    assert execution.last_close("AAA", dates[5]) is not None
    assert execution.open("AAA", dates[1]) == pytest.approx(
        float(prices[(prices["date"] == dates[1]) & (prices["ticker"] == "AAA")]["open"].iloc[0])
    )


def test_gate_rejects_empty_data(prices):
    with pytest.raises(DataError):
        TimeGate(prices, universe=["NOPE"])


def test_snapshot_missing_ticker_is_marked(prices):
    gate = TimeGate(prices, universe=["AAA", "BBB"])
    snap = gate.get(sorted(prices["date"].unique())[-1]).snapshot(["AAA", "BBB", "CCC"])
    assert snap.features["CCC"]["missing"] is True
    assert snap.features["CCC"]["stale"] is True
