"""Simple cross-sectional momentum: 12-month return skipping the latest month."""

from __future__ import annotations

import math

from baselines.base import StrategyContext, TargetDecision


class Momentum:
    name = "momentum"

    def __init__(self, lookback_days: int = 252, skip_days: int = 21, top_k: int = 5) -> None:
        self.lookback_days = int(lookback_days)
        self.skip_days = int(skip_days)
        self.top_k = int(top_k)

    def propose(self, ctx: StrategyContext) -> TargetDecision:
        scores = ctx.view.momentum(self.lookback_days, self.skip_days)
        scores = scores[[t for t in ctx.universe if t in scores.index]].dropna()
        if scores.empty:
            return TargetDecision(weights=None, reason="Insufficient history for momentum.")
        chosen = list(scores.sort_values(ascending=False).head(self.top_k).index)
        weight = 1.0 / len(chosen)
        return TargetDecision(
            weights={t: (weight if t in chosen else 0.0) for t in ctx.universe},
            reason=f"Top {len(chosen)} 12m-skip-1m momentum names, equal weighted.",
            flags=(f"selected:{','.join(chosen)}",),
        )
