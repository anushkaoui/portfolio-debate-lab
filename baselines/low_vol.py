"""Second simple signal: equal-weight the lowest-volatility names."""

from __future__ import annotations

from baselines.base import StrategyContext, TargetDecision


class LowVolatility:
    name = "low_vol"

    def __init__(self, lookback_days: int = 63, top_k: int = 5) -> None:
        self.lookback_days = int(lookback_days)
        self.top_k = int(top_k)

    def propose(self, ctx: StrategyContext) -> TargetDecision:
        vols = ctx.view.volatility(self.lookback_days)
        vols = vols[[t for t in ctx.universe if t in vols.index]].dropna()
        if vols.empty:
            return TargetDecision(weights=None, reason="Insufficient history for low-vol.")
        chosen = list(vols.sort_values(ascending=True).head(self.top_k).index)
        weight = 1.0 / len(chosen)
        return TargetDecision(
            weights={t: (weight if t in chosen else 0.0) for t in ctx.universe},
            reason=f"Lowest {len(chosen)} 3-month-vol names, equal weighted.",
            flags=(f"selected:{','.join(chosen)}",),
        )
