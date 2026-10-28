"""Performance metrics, block bootstrap, null comparison, deflated Sharpe and power."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def equity_returns(equity: pd.Series | Sequence[float]) -> pd.Series:
    s = pd.Series(equity, dtype=float).dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    return s.pct_change().dropna()


def annualized_return(returns: Sequence[float], periods_per_year: int = 52) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    return growth ** (periods_per_year / len(r)) - 1.0 if growth > 0 else -1.0


def annualized_volatility(returns: Sequence[float], periods_per_year: int = 52) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    return float(np.std(r, ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Sequence[float], risk_free_annual: float = 0.0, periods_per_year: int = 52
) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf
    sd = np.std(excess, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(excess) / sd * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float], risk_free_annual: float = 0.0, periods_per_year: int = 52
) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return float("nan")
    rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf
    downside = excess[excess < 0]
    dd = np.std(downside, ddof=1) if len(downside) > 1 else np.nan
    if not np.isfinite(dd) or dd == 0:
        return float("nan")
    return float(np.mean(excess) / dd * math.sqrt(periods_per_year))


def max_drawdown(equity: Sequence[float]) -> float:
    e = np.asarray(equity, dtype=float)
    if len(e) == 0:
        return float("nan")
    running = np.maximum.accumulate(e)
    dd = e / running - 1.0
    return float(np.min(dd))


def hit_rate(returns: Sequence[float]) -> float:
    r = np.asarray(returns, dtype=float)
    return float(np.mean(r > 0)) if len(r) else float("nan")


def summarize(
    equity: Sequence[float],
    *,
    turnovers: Sequence[float] | None = None,
    costs: Sequence[float] | None = None,
    initial_capital: float | None = None,
    risk_free_annual: float = 0.0,
    periods_per_year: int = 52,
) -> dict:
    e = pd.Series(equity, dtype=float).dropna()
    r = equity_returns(e)
    arr = r.to_numpy()
    out = {
        "n_periods": int(len(arr)),
        "total_return": float(e.iloc[-1] / e.iloc[0] - 1.0) if len(e) > 1 else float("nan"),
        "annualized_return": annualized_return(arr, periods_per_year),
        "annualized_volatility": annualized_volatility(arr, periods_per_year),
        "sharpe": sharpe_ratio(arr, risk_free_annual, periods_per_year),
        "sortino": sortino_ratio(arr, risk_free_annual, periods_per_year),
        "max_drawdown": max_drawdown(e.to_numpy()),
        "hit_rate": hit_rate(arr),
        "best_week": float(np.max(arr)) if len(arr) else float("nan"),
        "worst_week": float(np.min(arr)) if len(arr) else float("nan"),
    }
    if turnovers is not None and len(turnovers):
        out["mean_turnover"] = float(np.mean(turnovers))
        out["annualized_turnover"] = float(np.mean(turnovers) * periods_per_year)
    if costs is not None and len(costs):
        total_cost = float(np.sum(costs))
        out["total_cost"] = total_cost
        base = initial_capital or (float(e.iloc[0]) if len(e) else 1.0)
        out["cost_drag_of_initial"] = total_cost / base if base else float("nan")
    return out


def block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    block_length = max(1, min(block_length, n))
    n_blocks = math.ceil(n / block_length)
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate(
        [(np.arange(s, s + block_length) % n) for s in starts]
    )[:n]
    return idx


def bootstrap_ci(
    returns: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = sharpe_ratio,
    *,
    n_boot: int = 2000,
    block_length: int = 4,
    confidence: float = 0.95,
    seed: int = 20261028,
) -> dict:
    r = np.asarray(returns, dtype=float)
    if len(r) < 3:
        return {"estimate": float(stat_fn(r)) if len(r) else float("nan"), "low": float("nan"), "high": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_bootstrap_indices(len(r), block_length, rng)
        values[i] = stat_fn(r[idx])
    alpha = (1.0 - confidence) / 2.0
    values = values[np.isfinite(values)]
    return {
        "estimate": float(stat_fn(r)),
        "low": float(np.quantile(values, alpha)) if len(values) else float("nan"),
        "high": float(np.quantile(values, 1 - alpha)) if len(values) else float("nan"),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_diff(
    returns_a: Sequence[float],
    returns_b: Sequence[float],
    stat_fn: Callable[[np.ndarray], float] = sharpe_ratio,
    *,
    n_boot: int = 2000,
    block_length: int = 4,
    confidence: float = 0.95,
    seed: int = 20261028,
) -> dict:
    a = np.asarray(returns_a, dtype=float)
    b = np.asarray(returns_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3:
        return {"estimate": float("nan"), "low": float("nan"), "high": float("nan"), "n_boot": 0}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = block_bootstrap_indices(n, block_length, rng)
        diffs[i] = stat_fn(a[idx]) - stat_fn(b[idx])
    alpha = (1.0 - confidence) / 2.0
    diffs = diffs[np.isfinite(diffs)]
    estimate = float(stat_fn(a) - stat_fn(b))
    return {
        "estimate": estimate,
        "low": float(np.quantile(diffs, alpha)) if len(diffs) else float("nan"),
        "high": float(np.quantile(diffs, 1 - alpha)) if len(diffs) else float("nan"),
        "p_value_two_sided": float(
            min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
        )
        if len(diffs)
        else float("nan"),
        "n_boot": int(n_boot),
    }


def random_agent_percentile(strategy_sharpe: float, null_sharpes: Iterable[float]) -> float:
    arr = np.asarray([x for x in null_sharpes if np.isfinite(x)], dtype=float)
    if not len(arr) or not np.isfinite(strategy_sharpe):
        return float("nan")
    return float(100.0 * np.mean(arr < strategy_sharpe))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    trial_sharpes: Sequence[float],
    n_observations: int,
    *,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio."""
    trials = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
    n_trials = max(1, len(trials))
    if n_trials > 1:
        sr_std = float(np.std(trials, ddof=1))
    else:
        sr_std = 0.0
    gamma = 0.5772156649
    if n_trials > 1 and sr_std > 0:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max = sr_std * ((1.0 - gamma) * z1 + gamma * z2)
    else:
        expected_max = 0.0
    denom = math.sqrt(
        max(1e-12, 1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2)
    )
    if n_observations < 2:
        dsr = float("nan")
    else:
        z = (observed_sharpe - expected_max) * math.sqrt(n_observations - 1) / denom
        dsr = float(stats.norm.cdf(z))
    return {
        "observed_sharpe": float(observed_sharpe),
        "expected_max_sharpe": float(expected_max),
        "n_trials": n_trials,
        "deflated_sharpe_probability": dsr,
        "passes": bool(np.isfinite(dsr) and dsr > 0.95 and observed_sharpe > 0),
    }


def minimum_detectable_sharpe(
    n_weeks: int,
    *,
    weekly_volatility: float = 0.02,
    correlation: float = 0.90,
    alpha: float = 0.05,
    power: float = 0.80,
    n_sims: int = 2000,
    seed: int = 20261028,
) -> dict:
    """Simulate paired strategies and find the smallest annualized Sharpe gap
    detected at the requested power."""
    if n_weeks < 8:
        return {"n_weeks": n_weeks, "mde_sharpe": float("nan"), "note": "too few weeks"}
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, weekly_volatility, size=(n_sims, n_weeks))
    noise = rng.normal(0.0, weekly_volatility, size=(n_sims, n_weeks))
    other = correlation * base + math.sqrt(max(0.0, 1.0 - correlation**2)) * noise
    annual_factor = math.sqrt(52)
    grid = np.arange(0.05, 2.01, 0.05)
    target_power = power

    def power_for(delta_sharpe: float) -> float:
        shift = delta_sharpe * weekly_volatility / annual_factor
        a = base
        b = other + shift
        sa = a.mean(axis=1) / a.std(axis=1, ddof=1) * annual_factor
        sb = b.mean(axis=1) / b.std(axis=1, ddof=1) * annual_factor
        diff = sa - sb
        se = diff.std(ddof=1) / math.sqrt(n_sims)
        tcrit = stats.t.ppf(1 - alpha / 2, df=n_sims - 1)
        # Approximate power: probability the observed gap exceeds the critical
        # value under the alternative (normal approximation across sims).
        return float(stats.norm.cdf((diff.mean() - tcrit * se) / (se if se else 1.0)))

    # A more direct simulation: for each effect, count how often a paired
    # t-test across the n weeks rejects.
    def rejection_rate(delta_sharpe: float) -> float:
        shift = delta_sharpe * weekly_volatility / annual_factor
        b = other + shift
        diffs = b - base
        means = diffs.mean(axis=1)
        sds = diffs.std(axis=1, ddof=1)
        tstats = means / (sds / math.sqrt(n_weeks))
        crit = stats.t.ppf(1 - alpha / 2, df=n_weeks - 1)
        return float(np.mean(np.abs(tstats) > crit))

    previous = 0.0
    for delta in grid:
        rate = rejection_rate(float(delta))
        if rate >= target_power:
            return {
                "n_weeks": int(n_weeks),
                "weekly_volatility": float(weekly_volatility),
                "correlation": float(correlation),
                "alpha": float(alpha),
                "target_power": float(target_power),
                "simulated_power": float(rate),
                "mde_sharpe": float(delta),
                "n_sims": int(n_sims),
                "note": "Annualized Sharpe difference detectable with paired weekly returns.",
            }
        previous = rate
    return {
        "n_weeks": int(n_weeks),
        "mde_sharpe": float("nan"),
        "simulated_power_at_2": float(previous),
        "note": "MDE exceeds 2.0 Sharpe with this sample size.",
    }


def conviction_calibration(records: Sequence[dict]) -> dict:
    """records: [{'conviction': 0-1, 'realized_return': float}, ...]."""
    rows = [
        r
        for r in records
        if np.isfinite(float(r.get("conviction", np.nan)))
        and np.isfinite(float(r.get("realized_return", np.nan)))
    ]
    if len(rows) < 3:
        return {"n": len(rows), "spearman": float("nan"), "buckets": []}
    conv = np.array([float(r["conviction"]) for r in rows])
    ret = np.array([float(r["realized_return"]) for r in rows])
    rho = float(stats.spearmanr(conv, ret).statistic)
    buckets = []
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]:
        mask = (conv >= lo) & (conv < hi)
        if mask.any():
            buckets.append(
                {
                    "bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                    "n": int(mask.sum()),
                    "mean_realized_return": float(ret[mask].mean()),
                }
            )
    return {"n": len(rows), "spearman": rho, "buckets": buckets}
