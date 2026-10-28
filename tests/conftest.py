from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest


def business_dates(n: int, start: date = date(2023, 1, 2)) -> list[date]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@pytest.fixture
def prices() -> pd.DataFrame:
    dates = business_dates(40)
    rows = []
    for i, d in enumerate(dates):
        for j, ticker in enumerate(("AAA", "BBB")):
            base = 10.0 + j * 5.0 + i * 0.05 + (i % 3) * 0.01
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "open": base,
                    "high": base * 1.02,
                    "low": base * 0.98,
                    "close": base * 1.001,
                    "volume": 1_000_000 + i * 1000,
                    "dollar_volume": (1_000_000 + i * 1000) * base,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def zero_costs():
    from core.simulator import CostModel

    return CostModel(commission_bps=0.0, slippage_bps=0.0, cost_multiplier=1.0)
