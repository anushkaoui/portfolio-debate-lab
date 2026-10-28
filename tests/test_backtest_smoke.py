from __future__ import annotations

import math

import pytest

from core.backtest import BacktestSpec, run_backtest
from core.period_guard import PeriodError


def _spec(strategy: str, **kw) -> BacktestSpec:
    return BacktestSpec(
        strategy=strategy,
        period="dev",
        mode="mock",
        max_decisions=kw.pop("max_decisions", 3),
        write_ledger=False,
        write_trace=False,
        write_results=False,
        **kw,
    )


def test_equal_weight_buy_hold_smoke():
    result = run_backtest(_spec("equal_weight_buy_hold", max_decisions=4))
    assert result.decisions == 4
    assert result.metrics["fills"] == 20
    assert len(result.turnover) == 4
    assert len(result.equity) == 5
    assert result.metrics["provisional_periods"] is True
    assert math.isfinite(result.metrics["sharpe"])


def test_random_agent_smoke():
    result = run_backtest(_spec("random_agent", max_decisions=3, seed=11))
    assert result.decisions == 3
    assert result.metrics["fills"] > 0
    assert result.metrics["model"] == "code"


def test_mock_debate_smoke():
    result = run_backtest(_spec("debate", max_decisions=2, mock_seed=3))
    assert result.decisions == 2
    assert result.calls >= 2
    assert result.metrics["model"] == "deterministic-mock-v1"
    assert len(result.equity) == 3
    assert result.spec["strategy"] == "debate"
    assert result.metrics["conviction_calibration"]["n"] >= 0


def test_clean_test_requires_final(tmp_path):
    with pytest.raises(PeriodError):
        run_backtest(
            BacktestSpec(
                strategy="equal_weight_buy_hold",
                period="clean_test",
                write_ledger=False,
                write_trace=False,
                write_results=False,
            )
        )


def test_results_and_trace_can_be_written(tmp_path):
    spec = BacktestSpec(
        strategy="equal_weight_buy_hold",
        period="dev",
        max_decisions=2,
        write_ledger=False,
        write_trace=False,
        write_results=True,
        results_dir=str(tmp_path),
    )
    result = run_backtest(spec)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert result.run_id in files[0].name
