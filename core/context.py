"""Builds the numeric, anonymized context handed to agents.

All rolling features are computed inside the time gate. The builder never reads
price data directly. No calendar dates, real tickers, company names, absolute
prices or volumes are included in the primary context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from core.anonymize import Anonymizer
from core.config import sha256_obj
from core.timegate import FeatureSnapshot, TimeGate

FEATURE_KEYS = [
    "ret_1w",
    "ret_1m",
    "ret_3m",
    "vol_20d",
    "vol_60d",
    "drawdown_252d",
    "zscore_20d",
    "rel_volume_20d",
    "rank_ret_1m",
    "rank_vol_20d",
    "rank_ret_3m",
]


def _clean_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 6)


@dataclass(frozen=True)
class ContextPacket:
    context: dict[str, Any]
    context_hash: str
    as_of: date
    anonymized: bool
    code_to_ticker: dict[str, str]
    ticker_to_code: dict[str, str]
    snapshot: FeatureSnapshot

    def to_json(self) -> str:
        import json

        return json.dumps(self.context, sort_keys=True, separators=(",", ":"))


class ContextBuilder:
    def __init__(
        self,
        gate: TimeGate,
        anonymizer: Anonymizer,
        universe: Mapping[str, Any],
        risk_limits: Mapping[str, Any],
    ) -> None:
        self.gate = gate
        self.anonymizer = anonymizer
        self.tickers = list(universe["tickers"])
        self.companies = dict(universe.get("companies") or {})
        self.risk_limits = risk_limits
        self.stale_days = int(risk_limits.get("stale_data_days", 10))

    def build(
        self,
        as_of: date,
        *,
        portfolio_weights: Mapping[str, float] | None = None,
        cash_weight: float = 1.0,
        drawdown: float = 0.0,
        days_since_breaker: int | None = None,
        portfolio_value: float = 100000.0,
        identified: bool = False,
        snapshot: FeatureSnapshot | None = None,
    ) -> ContextPacket:
        view = self.gate.get(as_of)
        snap = snapshot or view.snapshot(self.tickers, stale_days=self.stale_days)
        weights = {str(k): float(v) for k, v in (portfolio_weights or {}).items()}

        max_pos = float(self.risk_limits.get("max_single_position", 0.10))
        max_group = float(self.risk_limits.get("max_group_exposure", 0.30))
        max_gross = float(self.risk_limits.get("max_gross_exposure", 1.0))
        max_turnover = float(self.risk_limits.get("max_turnover_per_rebalance", 0.5))
        max_adv_frac = float(self.risk_limits.get("max_order_fraction_of_adv", 0.10))

        vol_cfg = self.risk_limits.get("vol_scaled_cap") or {}
        target_vol = float(vol_cfg.get("target_annual_vol", 0.20))
        floor_ratio = float(vol_cfg.get("floor_ratio", 0.5))
        vol_enabled = bool(vol_cfg.get("enabled", True))

        codes = self.anonymizer.codes if not identified else self.tickers
        code_for = (lambda t: t) if identified else self.anonymizer.code
        features: dict[str, dict[str, Any]] = {}
        for ticker in self.tickers:
            code = code_for(ticker)
            raw = snap.features.get(ticker, {})
            row: dict[str, Any] = {}
            for key in FEATURE_KEYS:
                row[key] = _clean_number(raw.get(key))
            row["stale"] = bool(raw.get("stale", True))
            row["missing"] = bool(raw.get("missing", False))
            index_series = raw.get("price_index_26w") or []
            row["price_index_26w"] = [_clean_number(v) for v in index_series]
            if identified:
                row["company"] = self.companies.get(ticker, ticker)
                row["last_close"] = _clean_number(raw.get("last_close"))
                obs = raw.get("last_observation")
                if isinstance(obs, date):
                    row["last_observation"] = obs.isoformat()
            vol = raw.get("vol_20d")
            if vol_enabled and isinstance(vol, (int, float)) and float(vol) > 0:
                row["position_cap"] = round(
                    max_pos * min(1.0, max(floor_ratio, target_vol / float(vol))), 6
                )
            else:
                row["position_cap"] = max_pos
            row["group"] = (
                self.anonymizer.group_name(ticker)
                if identified
                else self.anonymizer.group_code(ticker)
            )
            # Liquidity headroom as a fraction of portfolio value.
            adv = float(snap.dollar_volume.get(ticker, 0.0) or 0.0)
            row["max_trade_weight"] = round(
                min(1.0, (max_adv_frac * adv) / max(portfolio_value, 1.0)), 6
            )
            features[code] = row

        portfolio_codes = {code_for(t): round(float(w), 6) for t, w in weights.items() if t in self.tickers}
        gross = sum(abs(v) for v in portfolio_codes.values())
        headroom = {
            "max_single_position": max_pos,
            "max_group_exposure": max_group,
            "max_gross_exposure": max_gross,
            "gross_exposure_left": round(max(0.0, max_gross - gross), 6),
            "max_turnover_per_rebalance": max_turnover,
            "drawdown_now": round(float(drawdown), 6),
            "days_since_breaker": days_since_breaker,
            "max_order_fraction_of_adv": max_adv_frac,
            "stale_data_days": self.stale_days,
        }

        context: dict[str, Any] = {
            "schema_version": "context-1.0",
            "time": {
                "as_of": "t-0" if not identified else as_of.isoformat(),
                "step": "1w",
                "return_windows": {"ret_1w": "t-1w", "ret_1m": "t-1m", "ret_3m": "t-3m"},
                "lookback_index_labels": ["t-26w", "t-0"],
            },
            "universe": codes,
            "features": features,
            "portfolio": {
                "weights": portfolio_codes,
                "cash_weight": round(float(cash_weight), 6),
                "drawdown": round(float(drawdown), 6),
            },
            "risk_headroom": headroom,
            "constraints": {
                "long_only": True,
                "max_gross_exposure": max_gross,
                "weights_are_fractions_of_portfolio": True,
                "absolute_prices_withheld": not identified,
            },
            "notes": [
                "All assets are anonymized unless identified mode is explicitly enabled.",
                "Use only the keys above. Missing or null values are unknown.",
            ],
        }
        if not identified:
            # The portfolio code map above is already anonymized; this guard
            # catches any accidental real-ticker key.
            self.anonymizer.assert_clean(context)
        context_hash = sha256_obj(context)
        if identified:
            code_to_ticker = {t: t for t in self.tickers}
            ticker_to_code = {t: t for t in self.tickers}
        else:
            code_to_ticker = dict(self.anonymizer.code_to_ticker())
            ticker_to_code = self.anonymizer.ticker_to_code()
        return ContextPacket(
            context=context,
            context_hash=context_hash,
            as_of=as_of,
            anonymized=not identified,
            code_to_ticker=code_to_ticker,
            ticker_to_code=ticker_to_code,
            snapshot=snap,
        )
