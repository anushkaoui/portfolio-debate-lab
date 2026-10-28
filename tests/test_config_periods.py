from __future__ import annotations

from datetime import date

import pytest

from core.config import config_hash, load_config, load_universe, stable_json
from core.period_guard import PeriodError, PeriodGuard
from core.periods import (
    decision_dates_for_period,
    load_period_table,
    weekly_decision_dates,
)


def test_universe_is_twenty_large_caps():
    universe = load_universe()
    assert len(universe["tickers"]) == 20
    assert set(universe["tickers"]) == set(universe["group_map"])
    assert all(universe["companies"][t] for t in universe["tickers"])


def test_config_hash_is_stable_and_order_insensitive():
    assert config_hash() == config_hash()
    assert stable_json({"a": 1, "b": 2}) == stable_json({"b": 2, "a": 1})


def test_period_table_exact_example_dates():
    table = load_period_table()
    p = table.periods
    assert p["warmup"].start == date(2019, 1, 1)
    assert p["warmup"].end == date(2019, 12, 31)
    assert p["stress_holdout"].start == date(2020, 1, 2)
    assert p["stress_holdout"].end == date(2022, 6, 30)
    assert p["dev"].start == date(2022, 7, 1)
    assert p["dev"].end == date(2023, 12, 29)
    assert p["precutoff_holdout"].start == date(2024, 1, 2)
    assert p["precutoff_holdout"].end == date(2024, 6, 28)
    assert p["embargo"].start == date(2024, 7, 1)
    assert p["embargo"].end == date(2024, 12, 31)
    assert p["clean_test"].start == date(2025, 1, 2)
    assert p["clean_test"].end == date(2026, 10, 23)
    assert p["forward"].start == date(2026, 10, 30)
    assert p["forward"].end is None
    assert table.clean_test_weeks == 95
    assert table.cutoff == date(2024, 6, 30)
    assert table.provisional is True
    assert table.cutoff_verified is False


def test_weekly_dates_are_fridays_or_prior_session():
    table = load_period_table()
    dates = decision_dates_for_period(table, "dev")
    assert dates
    assert len(dates) == len(set(dates))
    for d in dates:
        assert d.weekday() == 4 or d.weekday() < 4
        assert table.periods["dev"].start <= d <= table.periods["dev"].end


def test_guard_rejects_warmup_and_embargo():
    table = load_period_table()
    with pytest.raises(PeriodError):
        PeriodGuard(table, "warmup")
    with pytest.raises(PeriodError):
        PeriodGuard(table, "embargo")


def test_guard_rejects_clean_without_final():
    table = load_period_table()
    with pytest.raises(PeriodError, match="requires --final"):
        PeriodGuard(table, "clean_test")


def test_guard_dev_accepts_declared_dates():
    table = load_period_table()
    guard = PeriodGuard(table, "dev")
    for d in decision_dates_for_period(table, "dev")[:3]:
        assert guard.validate(d) == table.periods["dev"]
    with pytest.raises(PeriodError):
        guard.validate(table.periods["clean_test"].start)
