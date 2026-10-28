"""Baseline protocol."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Protocol

from core.timegate import FeatureSnapshot, GatedView


@dataclass(frozen=True)
class StrategyContext:
    as_of: date
    view: GatedView
    snapshot: FeatureSnapshot
    current_weights: Mapping[str, float]
    portfolio_value: float
    drawdown: float
    days_since_breaker: int | None
    universe: tuple[str, ...]


@dataclass(frozen=True)
class TargetDecision:
    weights: dict[str, float] | None
    reason: str
    flags: tuple[str, ...] = ()


class Strategy(Protocol):
    name: str

    def propose(self, ctx: StrategyContext) -> TargetDecision: ...
