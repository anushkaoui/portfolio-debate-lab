from __future__ import annotations

import math

import numpy as np
import pytest

from core.metrics import (
    annualized_return,
    annualized_volatility,
    block_bootstrap_indices,
    bootstrap_ci,
    conviction_calibration,
    deflated_sharpe_ratio,
    equity_returns,
    max_drawdown,
    minimum_detectable_sharpe,
    paired_bootstrap_diff,
    random_agent_percentile,
    sharpe_ratio,
    summarize,
)


def test_basic_metrics():
    equity = [100.0, 101.0, 100.0, 102.0]
    r = equity_returns(equity).to_numpy()
    assert len(r) == 3
    assert annualized_return(r) > -1
    assert annualized_volatility(r) > 0
    assert math.isfinite(sharpe_ratio(r))
    assert max_drawdown(equity) < 0
    out = summarize(equity, turnovers=[0.1], costs=[1.0], initial_capital=100.0)
    assert out["n_periods"] == 3
    assert out["total_cost"] == 1.0


def test_bootstrap_is_seeded_and_bounded():
    rng = np.random.default_rng(1)
    r = rng.normal(0.002, 0.02, 80)
    a = bootstrap_ci(r, n_boot=300, seed=7)
    b = bootstrap_ci(r, n_boot=300, seed=7)
    assert a == b
    assert a["low"] <= a["high"]
    idx = block_bootstrap_indices(10, 3, np.random.default_rng(3))
    assert len(idx) == 10
    assert idx.min() >= 0 and idx.max() < 10


def test_paired_diff_and_null_percentile():
    rng = np.random.default_rng(2)
    a = rng.normal(0.002, 0.02, 100)
    b = rng.normal(0.001, 0.02, 100)
    diff = paired_bootstrap_diff(a, b, n_boot=200, seed=4)
    assert -1.0 <= diff["p_value_two_sided"] <= 1.0
    assert random_agent_percentile(1.0, [0.0, 0.5, 1.5]) == pytest.approx(66.666, rel=1e-3)
    assert math.isnan(random_agent_percentile(float("nan"), [0.0]))


def test_conviction_calibration():
    records = [
        {"conviction": 0.2, "realized_return": -0.01},
        {"conviction": 0.5, "realized_return": 0.0},
        {"conviction": 0.9, "realized_return": 0.02},
    ]
    out = conviction_calibration(records)
    assert out["n"] == 3
    assert math.isfinite(out["spearman"])


def test_deflated_sharpe_and_power():
    dsr = deflated_sharpe_ratio(1.0, [0.1, 0.2, 0.3], 100)
    assert set(dsr) >= {"deflated_sharpe_probability", "n_trials", "passes"}
    mde = minimum_detectable_sharpe(95, n_sims=300, seed=1)
    assert mde["mde_sharpe"] > 0
    assert mde["n_weeks"] == 95
    short = minimum_detectable_sharpe(4)
    assert math.isnan(short["mde_sharpe"])
