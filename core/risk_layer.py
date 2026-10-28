"""Deterministic hard risk layer.

This module reads numbers only. It has no prompt, rationale or free-text input.
Agents cannot modify its configuration, and it has the final say on weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from numbers import Real
from typing import Mapping

EPS = 1e-9


class RiskInputError(TypeError):
    pass


@dataclass(frozen=True)
class RiskConfig:
    max_single_position: float = 0.10
    max_group_exposure: float = 0.30
    max_gross_exposure: float = 1.00
    allow_shorting: bool = False
    min_cash_weight: float = 0.0
    max_turnover_per_rebalance: float = 0.50
    max_order_fraction_of_adv: float = 0.10
    adv_lookback_days: int = 20
    stale_data_days: int = 10
    vol_scaled_enabled: bool = True
    target_annual_vol: float = 0.20
    vol_floor_ratio: float = 0.50
    drawdown_enabled: bool = True
    drawdown_reduce_threshold: float = -0.15
    drawdown_de_risk_fraction: float = 0.40
    drawdown_halt_threshold: float = -0.25
    cooloff_trading_days: int = 20

    @classmethod
    def from_dict(cls, cfg: Mapping) -> "RiskConfig":
        vol = cfg.get("vol_scaled_cap") or {}
        dd = cfg.get("drawdown_breaker") or {}
        return cls(
            max_single_position=float(cfg.get("max_single_position", 0.10)),
            max_group_exposure=float(cfg.get("max_group_exposure", 0.30)),
            max_gross_exposure=float(cfg.get("max_gross_exposure", 1.0)),
            allow_shorting=bool(cfg.get("allow_shorting", False)),
            min_cash_weight=float(cfg.get("min_cash_weight", 0.0)),
            max_turnover_per_rebalance=float(cfg.get("max_turnover_per_rebalance", 0.5)),
            max_order_fraction_of_adv=float(cfg.get("max_order_fraction_of_adv", 0.10)),
            adv_lookback_days=int(cfg.get("adv_lookback_days", 20)),
            stale_data_days=int(cfg.get("stale_data_days", 10)),
            vol_scaled_enabled=bool(vol.get("enabled", True)),
            target_annual_vol=float(vol.get("target_annual_vol", 0.20)),
            vol_floor_ratio=float(vol.get("floor_ratio", 0.50)),
            drawdown_enabled=bool(dd.get("enabled", True)),
            drawdown_reduce_threshold=float(dd.get("reduce_threshold", -0.15)),
            drawdown_de_risk_fraction=float(dd.get("de_risk_fraction", 0.40)),
            drawdown_halt_threshold=float(dd.get("halt_threshold", -0.25)),
            cooloff_trading_days=int(dd.get("cooloff_trading_days", 20)),
        )

    def validate(self) -> None:
        if not (0 < self.max_single_position <= 1):
            raise ValueError("max_single_position must be in (0, 1]")
        if not (0 < self.max_group_exposure <= 1):
            raise ValueError("max_group_exposure must be in (0, 1]")
        if not (0 <= self.max_gross_exposure <= 1):
            raise ValueError("max_gross_exposure must be in [0, 1]")
        if not (0 <= self.min_cash_weight < 1):
            raise ValueError("min_cash_weight must be in [0, 1)")
        if not (0 <= self.max_turnover_per_rebalance <= 1):
            raise ValueError("max_turnover_per_rebalance must be in [0, 1]")
        if not (0 <= self.max_order_fraction_of_adv <= 1):
            raise ValueError("max_order_fraction_of_adv must be in [0, 1]")
        if not (0 <= self.vol_floor_ratio <= 1):
            raise ValueError("vol_floor_ratio must be in [0, 1]")


@dataclass(frozen=True)
class Intervention:
    rule: str
    ticker: str
    requested: float
    applied: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "ticker": self.ticker,
            "requested": round(float(self.requested), 8),
            "applied": round(float(self.applied), 8),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RiskVerdict:
    approved_weights: dict[str, float]
    interventions: tuple[Intervention, ...]
    halted: bool
    cash_weight: float
    gross_exposure: float
    drawdown: float

    def to_dict(self) -> dict:
        return {
            "approved_weights": {k: round(v, 8) for k, v in self.approved_weights.items()},
            "interventions": [i.to_dict() for i in self.interventions],
            "halted": self.halted,
            "cash_weight": round(self.cash_weight, 8),
            "gross_exposure": round(self.gross_exposure, 8),
            "drawdown": round(self.drawdown, 8),
        }


@dataclass(frozen=True)
class RiskInputs:
    as_of: date
    target_weights: Mapping[str, float]
    current_weights: Mapping[str, float]
    portfolio_value: float
    prices: Mapping[str, float]
    volatilities: Mapping[str, float]
    adv_notional: Mapping[str, float]
    stale_tickers: frozenset[str] = frozenset()
    drawdown: float = 0.0
    days_since_breaker: int | None = None

    def __post_init__(self) -> None:
        for name, mapping in (
            ("target_weights", self.target_weights),
            ("current_weights", self.current_weights),
            ("prices", self.prices),
            ("volatilities", self.volatilities),
            ("adv_notional", self.adv_notional),
        ):
            for key, value in mapping.items():
                if not isinstance(key, str):
                    raise RiskInputError(f"{name} keys must be ticker strings")
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise RiskInputError(
                        f"{name}[{key!r}] must be numeric; prose is not accepted"
                    )
                if not math.isfinite(float(value)):
                    raise RiskInputError(f"{name}[{key!r}] must be finite")
        if not isinstance(self.portfolio_value, Real) or not math.isfinite(
            float(self.portfolio_value)
        ):
            raise RiskInputError("portfolio_value must be a finite number")
        if not isinstance(self.drawdown, Real) or not math.isfinite(float(self.drawdown)):
            raise RiskInputError("drawdown must be a finite number")


@dataclass
class _Working:
    weights: dict[str, float]
    interventions: list[Intervention] = field(default_factory=list)
    halted: bool = False

    def log(self, rule: str, ticker: str, requested: float, applied: float, detail: str = "") -> None:
        if abs(requested - applied) > 1e-10:
            self.interventions.append(
                Intervention(rule, ticker, float(requested), float(applied), detail)
            )


class RiskLayer:
    def __init__(
        self,
        config: RiskConfig,
        universe: list[str],
        group_map: Mapping[str, str],
    ) -> None:
        config.validate()
        self.config = config
        self.universe = sorted(set(universe))
        self.group_map = {str(k): str(v) for k, v in group_map.items()}

    @classmethod
    def from_config(
        cls,
        cfg: Mapping,
        universe: list[str],
        group_map: Mapping[str, str],
    ) -> "RiskLayer":
        return cls(RiskConfig.from_dict(cfg), universe, group_map)

    # ---- helpers -------------------------------------------------------
    def _cap_for(self, ticker: str, inputs: RiskInputs) -> float:
        cap = self.config.max_single_position
        if self.config.vol_scaled_enabled:
            vol = float(inputs.volatilities.get(ticker, 0.0) or 0.0)
            if vol > 0:
                ratio = min(1.0, max(self.config.vol_floor_ratio, self.config.target_annual_vol / vol))
                cap *= ratio
        return max(0.0, min(cap, self.config.max_single_position))

    def _gross_cap(self, inputs: RiskInputs) -> tuple[float, bool]:
        cap = min(self.config.max_gross_exposure, 1.0 - self.config.min_cash_weight)
        halted = False
        if self.config.drawdown_enabled and inputs.drawdown <= self.config.drawdown_reduce_threshold:
            cap *= max(0.0, 1.0 - self.config.drawdown_de_risk_fraction)
            if inputs.drawdown <= self.config.drawdown_halt_threshold:
                cap *= 0.5
                halted = True
        return max(0.0, cap), halted

    def _scale_to_gross(self, w: dict[str, float], cap: float) -> float:
        gross = sum(max(0.0, v) for v in w.values())
        if gross > cap + EPS and gross > 0:
            factor = cap / gross
            for ticker in w:
                w[ticker] *= factor
            return factor
        return 1.0

    def _apply_group_caps(self, w: dict[str, float], work: _Working) -> bool:
        changed = False
        groups: dict[str, list[str]] = {}
        for ticker in self.universe:
            group = self.group_map.get(ticker, ticker)
            groups.setdefault(group, []).append(ticker)
        for group, members in groups.items():
            exposure = sum(max(0.0, w.get(t, 0.0)) for t in members)
            if exposure > self.config.max_group_exposure + EPS and exposure > 0:
                factor = self.config.max_group_exposure / exposure
                for ticker in members:
                    before = w.get(ticker, 0.0)
                    w[ticker] = before * factor
                    if abs(before - w[ticker]) > 1e-12:
                        work.log(
                            "max_group_exposure",
                            ticker,
                            before,
                            w[ticker],
                            f"group={group}; group_after={exposure * factor:.6f}",
                        )
                changed = True
        return changed

    def _apply_turnover(self, w: dict[str, float], current: dict[str, float], work: _Working) -> bool:
        deltas = {t: w.get(t, 0.0) - current.get(t, 0.0) for t in self.universe}
        sells = sum(max(0.0, -d) for d in deltas.values())
        buys = sum(max(0.0, d) for d in deltas.values())
        turnover = 0.5 * (sells + buys)
        limit = self.config.max_turnover_per_rebalance
        if turnover <= limit + EPS or turnover <= 0:
            return False
        # Preserve risk-reducing sells and cut purchases first. If even sells
        # exceed the budget (only possible with gross > 1), scale sells too.
        allowed_total = 2.0 * limit
        sell_factor = 1.0 if sells <= allowed_total else allowed_total / sells
        allowed_buys = max(0.0, allowed_total - sells * sell_factor)
        buy_factor = 1.0 if buys <= allowed_buys or buys <= 0 else allowed_buys / buys
        changed = False
        for ticker, delta in deltas.items():
            before = w.get(ticker, 0.0)
            factor = sell_factor if delta < 0 else buy_factor
            applied_delta = delta * factor
            new = current.get(ticker, 0.0) + applied_delta
            new = max(0.0, new)
            if not self.config.allow_shorting:
                new = max(0.0, new)
            if abs(new - before) > 1e-12:
                changed = True
            w[ticker] = new
            work.log(
                "max_turnover",
                ticker,
                before,
                new,
                f"requested_turnover={turnover:.6f}; sell_factor={sell_factor:.6f}; "
                f"buy_factor={buy_factor:.6f}",
            )
        return changed

    def _apply_liquidity(
        self, w: dict[str, float], current: dict[str, float], inputs: RiskInputs, work: _Working
    ) -> bool:
        changed = False
        if inputs.portfolio_value <= 0:
            return False
        for ticker in self.universe:
            adv = float(inputs.adv_notional.get(ticker, 0.0) or 0.0)
            if adv <= 0:
                continue
            max_weight_delta = (
                self.config.max_order_fraction_of_adv * adv / inputs.portfolio_value
            )
            delta = w.get(ticker, 0.0) - current.get(ticker, 0.0)
            # Risk-reducing sells are never blocked by ADV; only purchases are.
            if delta > max_weight_delta + EPS:
                before = w.get(ticker, 0.0)
                safe_delta = max(0.0, max_weight_delta - 1e-12)
                w[ticker] = current.get(ticker, 0.0) + math.copysign(safe_delta, delta)
                w[ticker] = max(0.0, w[ticker])
                work.log(
                    "liquidity_limit",
                    ticker,
                    before,
                    w[ticker],
                    f"adv_fraction={self.config.max_order_fraction_of_adv}; "
                    f"max_weight_delta={max_weight_delta:.8f}",
                )
                changed = True
        return changed

    # ---- public API ----------------------------------------------------
    def apply(self, inputs: RiskInputs) -> RiskVerdict:
        current = {t: max(0.0, float(inputs.current_weights.get(t, 0.0))) for t in self.universe}
        # Existing holdings above the single-name cap are reduced only via sells,
        # but a hold-level target is not forced down when the target equals the
        # current weight in this call. New targets are capped.
        w: dict[str, float] = {
            t: max(0.0, float(inputs.target_weights.get(t, 0.0))) for t in self.universe
        }
        work = _Working(weights=w)

        unknown = set(inputs.target_weights) - set(self.universe)
        for ticker in sorted(unknown):
            work.log("unknown_ticker", str(ticker), float(inputs.target_weights[ticker]), 0.0)

        if not self.config.allow_shorting:
            for ticker, value in list(inputs.target_weights.items()):
                if ticker in w and value < 0:
                    work.log("no_shorting", ticker, float(value), 0.0)
                    w[ticker] = 0.0

        # Stale or missing data: freezes the name at its current weight.
        for ticker in sorted(inputs.stale_tickers):
            if ticker not in self.universe:
                continue
            requested = w[ticker]
            w[ticker] = current[ticker]
            work.log("stale_data", ticker, requested, w[ticker], "hold_until_data_current")

        # Iterate interactions until stable. The rules are contractions; the
        # second call on an approved target is a no-op (idempotence).
        for _ in range(30):
            changed = False
            # Single-name and volatility caps.
            for ticker in self.universe:
                cap = self._cap_for(ticker, inputs)
                if w[ticker] > cap + EPS:
                    before = w[ticker]
                    w[ticker] = cap
                    work.log("max_position_or_vol_cap", ticker, before, cap)
                    changed = True
            # Group caps.
            changed = self._apply_group_caps(w, work) or changed
            # Drawdown / cool-off. Absolute gross cap keeps the rule idempotent.
            gross_cap, halted = self._gross_cap(inputs)
            work.halted = work.halted or halted
            if halted or inputs.days_since_breaker is not None:
                if inputs.days_since_breaker is None or inputs.days_since_breaker < self.config.cooloff_trading_days:
                    for ticker in self.universe:
                        if w[ticker] > current[ticker] + EPS:
                            before = w[ticker]
                            w[ticker] = current[ticker]
                            work.log(
                                "cooloff_no_new_buys",
                                ticker,
                                before,
                                w[ticker],
                                f"days_since_breaker={inputs.days_since_breaker}",
                            )
                            changed = True
            before_gross = sum(w.values())
            factor = self._scale_to_gross(w, gross_cap)
            if factor < 1.0:
                work.log(
                    "drawdown_gross_cap" if inputs.drawdown <= self.config.drawdown_reduce_threshold else "gross_exposure",
                    "*",
                    before_gross,
                    sum(w.values()),
                    f"gross_cap={gross_cap:.6f}; factor={factor:.6f}",
                )
                changed = True
            # Liquidity then turnover.
            changed = self._apply_liquidity(w, current, inputs, work) or changed
            changed = self._apply_turnover(w, current, work) or changed
            if not changed:
                break

        # Final safety pass: single-name cap, group cap, gross cap (no logging
        # loops). These are protections against pathological numeric inputs.
        for ticker in self.universe:
            w[ticker] = max(0.0, min(w.get(ticker, 0.0), self._cap_for(ticker, inputs)))
        self._apply_group_caps(w, work)
        gross_cap, _ = self._gross_cap(inputs)
        self._scale_to_gross(w, gross_cap)
        # Never exceed the cash floor after the final scaling.
        gross = sum(w.values())
        max_gross = min(self.config.max_gross_exposure, 1.0 - self.config.min_cash_weight)
        if gross > max_gross + EPS:
            self._scale_to_gross(w, max_gross)
        # Cutting purchases after the final cap pass restores the turnover
        # budget without loosening any cap or increasing any position.
        self._apply_turnover(w, current, work)
        cash_weight = max(0.0, 1.0 - sum(w.values()))
        return RiskVerdict(
            approved_weights={t: float(max(0.0, w.get(t, 0.0))) for t in self.universe},
            interventions=tuple(work.interventions),
            halted=bool(work.halted),
            cash_weight=cash_weight,
            gross_exposure=sum(w.values()),
            drawdown=float(inputs.drawdown),
        )

    # ---- independent checks used by tests and evaluation ---------------
    def breaches(self, verdict: RiskVerdict, inputs: RiskInputs | None = None) -> list[str]:
        problems: list[str] = []
        w = verdict.approved_weights
        tol = 1e-5
        for ticker in self.universe:
            value = w.get(ticker, 0.0)
            if value < -tol:
                problems.append(f"short:{ticker}:{value}")
            cap = self._cap_for(ticker, inputs) if inputs else self.config.max_single_position
            if value > cap + tol:
                problems.append(f"single:{ticker}:{value}>{cap}")
        groups: dict[str, float] = {}
        for ticker in self.universe:
            groups[self.group_map.get(ticker, ticker)] = (
                groups.get(self.group_map.get(ticker, ticker), 0.0) + max(0.0, w.get(ticker, 0.0))
            )
        for group, value in groups.items():
            if value > self.config.max_group_exposure + tol:
                problems.append(f"group:{group}:{value}")
        gross = sum(max(0.0, v) for v in w.values())
        if gross > self.config.max_gross_exposure + tol:
            problems.append(f"gross:{gross}")
        if 1.0 - gross < self.config.min_cash_weight - tol:
            problems.append(f"cash_floor:{1.0 - gross}")
        if inputs is not None:
            current = {t: max(0.0, float(inputs.current_weights.get(t, 0.0))) for t in self.universe}
            turnover = 0.5 * sum(abs(w.get(t, 0.0) - current.get(t, 0.0)) for t in self.universe)
            if turnover > self.config.max_turnover_per_rebalance + tol:
                problems.append(f"turnover:{turnover}")
            for ticker in self.universe:
                adv = float(inputs.adv_notional.get(ticker, 0.0) or 0.0)
                if adv > 0 and inputs.portfolio_value > 0:
                    max_delta = self.config.max_order_fraction_of_adv * adv / inputs.portfolio_value
                    delta = w.get(ticker, 0.0) - current.get(ticker, 0.0)
                    if delta > max_delta + tol:
                        problems.append(f"liquidity:{ticker}")
            gross_cap, _ = self._gross_cap(inputs)
            if gross > gross_cap + tol:
                problems.append(f"drawdown_cap:{gross}>{gross_cap}")
        return problems
