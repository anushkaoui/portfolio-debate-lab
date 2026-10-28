"""The single as-of access path for all agent-visible market data.

Everything that feeds a decision goes through `TimeGate.get(as_of)`. The gate
returns a `GatedView` whose slice methods raise if asked for a date after
`as_of`. Execution prices for the simulator are deliberately a separate object
(`ExecutionPrices`) that is never handed to context builders or agents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


class LookAheadError(RuntimeError):
    pass


class DataError(ValueError):
    pass


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def load_prices(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Price file not found: {path}. Run scripts/fetch_data.py for dev data."
        )
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise DataError(f"Unsupported price file format: {path.suffix}")

    df = df.copy()
    if "date" not in df.columns and df.index.name in {"date", "Date"}:
        df = df.reset_index()
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataError(f"Price data missing columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    if "dollar_volume" not in df.columns:
        df["dollar_volume"] = df["close"] * df["volume"]
    df = df.dropna(subset=["date", "ticker", "open", "close", "volume"])
    df = df[df["ticker"] != ""]
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    if df.empty:
        raise DataError(f"No usable rows in {path}")
    return df


def load_price_file(path: str | Path | None = None, default: str = "data/dev/prices.parquet") -> pd.DataFrame:
    from core.config import ROOT

    return load_prices(Path(path) if path else ROOT / default)


class ExecutionPrices:
    """Full price history for the simulator. Never pass this to agents."""

    def __init__(self, prices: pd.DataFrame) -> None:
        self._df = prices.copy()
        self._wide_open = self._df.pivot(index="date", columns="ticker", values="open").sort_index()
        self._wide_close = self._df.pivot(index="date", columns="ticker", values="close").sort_index()
        self._wide_dollar_volume = (
            self._df.pivot(index="date", columns="ticker", values="dollar_volume").sort_index()
        )
        self.dates = list(self._wide_close.index)
        self.dates_set = set(self.dates)

    def has_date(self, day: date) -> bool:
        return _to_date(day) in self.dates_set

    def next_trading_day(self, day: date) -> date:
        day = _to_date(day)
        for d in self.dates:
            if d > day:
                return d
        raise LookAheadError(f"No trading day after {day.isoformat()} in execution data")

    def previous_trading_day(self, day: date) -> date:
        day = _to_date(day)
        prev = None
        for d in self.dates:
            if d >= day:
                break
            prev = d
        if prev is None:
            raise LookAheadError(f"No trading day before {day.isoformat()} in execution data")
        return prev

    def open(self, ticker: str, day: date) -> float | None:
        try:
            v = self._wide_open.at[_to_date(day), ticker]
        except KeyError:
            return None
        return None if pd.isna(v) else float(v)

    def close(self, ticker: str, day: date) -> float | None:
        try:
            v = self._wide_close.at[_to_date(day), ticker]
        except KeyError:
            return None
        return None if pd.isna(v) else float(v)

    def last_close(self, ticker: str, day: date) -> float | None:
        if ticker not in self._wide_close.columns:
            return None
        s = self._wide_close[ticker].loc[: _to_date(day)].dropna()
        if s.empty:
            return None
        return float(s.iloc[-1])

    def adv_notional(self, ticker: str, day: date, lookback: int = 20) -> float | None:
        day = _to_date(day)
        try:
            s = self._wide_dollar_volume.loc[:day, ticker].dropna().tail(lookback)
        except KeyError:
            return None
        if s.empty:
            return None
        return float(s.mean())

    def closes_on(self, day: date) -> dict[str, float]:
        day = _to_date(day)
        if day not in self._wide_close.index:
            return {}
        row = self._wide_close.loc[day].dropna()
        return {str(k): float(v) for k, v in row.items()}

    def opens_on(self, day: date) -> dict[str, float]:
        day = _to_date(day)
        if day not in self._wide_open.index:
            return {}
        row = self._wide_open.loc[day].dropna()
        return {str(k): float(v) for k, v in row.items()}

    def last_date(self) -> date:
        return self.dates[-1]


@dataclass(frozen=True)
class FeatureSnapshot:
    as_of: date
    closes: dict[str, float]
    features: dict[str, dict[str, float | bool | str]]
    dollar_volume: dict[str, float]
    trading_dates: tuple[date, ...]

    def stale_tickers(self, stale_days: int) -> set[str]:
        stale: set[str] = set()
        for ticker, feats in self.features.items():
            if bool(feats.get("stale", False)):
                stale.add(ticker)
        return stale


class GatedView:
    def __init__(self, frame: pd.DataFrame, as_of: date) -> None:
        self._frame = frame
        self.as_of = _to_date(as_of)
        self._cache: dict[str, pd.DataFrame] = {}

    def _slice(self, end: date | None = None) -> pd.DataFrame:
        end = _to_date(end or self.as_of)
        if end > self.as_of:
            raise LookAheadError(
                f"Attempted to read {end.isoformat()} with as_of={self.as_of.isoformat()}"
            )
        key = end.isoformat()
        if key not in self._cache:
            self._cache[key] = self._frame[self._frame["date"] <= end].copy()
        return self._cache[key]

    def rows(self) -> pd.DataFrame:
        return self._slice().copy()

    def trading_dates(self, lookback: int | None = None) -> list[date]:
        dates = sorted(self._slice()["date"].unique())
        if lookback:
            dates = dates[-lookback:]
        return [d if isinstance(d, date) else pd.Timestamp(d).date() for d in dates]

    def close_panel(self, lookback: int | None = None) -> pd.DataFrame:
        df = self._slice()
        if lookback:
            keep = set(self.trading_dates(lookback))
            df = df[df["date"].isin(keep)]
        panel = df.pivot(index="date", columns="ticker", values="close").sort_index()
        return panel

    def open_panel(self) -> pd.DataFrame:
        return self._slice().pivot(index="date", columns="ticker", values="open").sort_index()

    def volume_panel(self, lookback: int | None = None) -> pd.DataFrame:
        df = self._slice()
        if lookback:
            keep = set(self.trading_dates(lookback))
            df = df[df["date"].isin(keep)]
        return df.pivot(index="date", columns="ticker", values="volume").sort_index()

    def dollar_volume_panel(self, lookback: int = 20) -> pd.DataFrame:
        df = self._slice()
        keep = set(self.trading_dates(lookback))
        df = df[df["date"].isin(keep)]
        return df.pivot(index="date", columns="ticker", values="dollar_volume").sort_index()

    def snapshot(
        self,
        universe: Iterable[str],
        *,
        stale_days: int = 10,
        price_index_weeks: int = 26,
    ) -> FeatureSnapshot:
        panel = self.close_panel(lookback=252)
        volume = self.volume_panel(lookback=252)
        dv = self.dollar_volume_panel(lookback=20)
        if panel.empty:
            raise DataError(f"No data at or before {self.as_of.isoformat()}")

        features: dict[str, dict[str, float | bool | str]] = {}
        closes: dict[str, float] = {}
        dollar_volume: dict[str, float] = {}
        for ticker in universe:
            if ticker not in panel.columns:
                features[ticker] = {"stale": True, "missing": True}
                continue
            s = panel[ticker].dropna()
            if s.empty:
                features[ticker] = {"stale": True, "missing": True}
                continue
            last_date = _to_date(s.index[-1])
            last = float(s.iloc[-1])
            closes[ticker] = last
            dollar_volume[ticker] = (
                float(dv[ticker].dropna().mean()) if ticker in dv.columns and not dv[ticker].dropna().empty else 0.0
            )

            def ret(periods: int) -> float:
                if len(s) <= periods:
                    return float("nan")
                base = float(s.iloc[-1 - periods])
                return last / base - 1.0 if base else float("nan")

            daily = s.pct_change().dropna()
            vol20 = float(daily.tail(20).std(ddof=0) * math.sqrt(252)) if len(daily) >= 20 else float("nan")
            vol60 = float(daily.tail(60).std(ddof=0) * math.sqrt(252)) if len(daily) >= 60 else float("nan")
            window252 = s.tail(252)
            drawdown = last / float(window252.max()) - 1.0 if len(window252) else float("nan")
            r20 = daily.tail(20)
            zscore = (
                float((r20.iloc[-1] - r20.mean()) / r20.std(ddof=0))
                if len(r20) >= 20 and r20.std(ddof=0) > 0
                else float("nan")
            )
            vols = volume[ticker].dropna().tail(20) if ticker in volume.columns else pd.Series(dtype=float)
            rel_volume = (
                float(vols.iloc[-1] / vols.mean())
                if len(vols) >= 5 and vols.mean() > 0
                else float("nan")
            )
            sampled = s.iloc[::5].tail(price_index_weeks)
            if sampled.empty or sampled.iloc[0] == 0:
                index_series: list[float] = []
            else:
                index_series = [round(float(v / sampled.iloc[0] * 100.0), 4) for v in sampled]

            features[ticker] = {
                "last_close": last,
                "last_observation": last_date,
                "ret_1w": ret(5),
                "ret_1m": ret(21),
                "ret_3m": ret(63),
                "vol_20d": vol20,
                "vol_60d": vol60,
                "drawdown_252d": drawdown,
                "zscore_20d": zscore,
                "rel_volume_20d": rel_volume,
                "price_index_26w": index_series,
                "stale": (self.as_of - last_date).days > stale_days,
                "missing": False,
            }

        self._add_cross_sectional_ranks(features)
        return FeatureSnapshot(
            as_of=self.as_of,
            closes=closes,
            features=features,
            dollar_volume=dollar_volume,
            trading_dates=tuple(self.trading_dates()),
        )

    @staticmethod
    def _add_cross_sectional_ranks(features: dict[str, dict]) -> None:
        for key in ["ret_1m", "vol_20d", "ret_3m"]:
            vals = {
                t: float(f[key])
                for t, f in features.items()
                if key in f and isinstance(f.get(key), (int, float)) and not math.isnan(float(f[key]))
            }
            if not vals:
                continue
            ordered = sorted(vals, key=vals.get)
            denom = max(len(ordered) - 1, 1)
            for rank, ticker in enumerate(ordered):
                features[ticker][f"rank_{key}"] = round(rank / denom, 6)

    def momentum(self, lookback: int = 252, skip: int = 21) -> pd.Series:
        panel = self.close_panel(lookback + skip + 1)
        if len(panel) < skip + 2:
            return pd.Series(dtype=float)
        end = panel.iloc[-1 - skip]
        start = panel.iloc[0]
        return (end / start - 1.0).dropna()

    def volatility(self, lookback: int = 63) -> pd.Series:
        panel = self.close_panel(lookback + 1)
        return panel.pct_change().std(ddof=0) * math.sqrt(252)

    def assert_no_future(self) -> None:
        sliced = self._slice()
        if not sliced.empty and max(sliced["date"]) > self.as_of:
            raise LookAheadError("Gated view contains future rows")


class TimeGate:
    def __init__(
        self,
        prices: pd.DataFrame,
        *,
        universe: Iterable[str] | None = None,
        max_visible_date: date | None = None,
    ) -> None:
        df = prices.copy()
        if universe is not None:
            df = df[df["ticker"].isin(list(universe))]
        if max_visible_date is not None:
            df = df[df["date"] <= _to_date(max_visible_date)]
        if df.empty:
            raise DataError("TimeGate has no data after applying its universe/date filter")
        self._frame = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.universe = sorted(self._frame["ticker"].unique().tolist())
        self.first_date = min(self._frame["date"])
        self.last_date = max(self._frame["date"])

    def get(self, as_of: date) -> GatedView:
        return GatedView(self._frame, _to_date(as_of))

    def execution_prices(self) -> ExecutionPrices:
        """Separate object for the simulator. Do not pass to agents."""
        return ExecutionPrices(self._frame)

    def __repr__(self) -> str:
        return (
            f"TimeGate({len(self.universe)} tickers, "
            f"{self.first_date} to {self.last_date})"
        )
