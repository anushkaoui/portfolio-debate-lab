"""Equal-weight baselines."""

from __future__ import annotations

from baselines.base import StrategyContext, TargetDecision


class EqualWeightBuyHold:
    name = "equal_weight_buy_hold"

    def __init__(self) -> None:
        self._invested = False

    def propose(self, ctx: StrategyContext) -> TargetDecision:
        if not self._invested:
            self._invested = True
            n = len(ctx.universe)
            return TargetDecision(
                weights={t: 1.0 / n for t in ctx.universe},
                reason="Initial equal-weight purchase; hold thereafter.",
            )
        return TargetDecision(weights=None, reason="Buy-and-hold; no rebalance.")


class EqualWeightRebalanced:
    name = "equal_weight_rebalanced"

    def propose(self, ctx: StrategyContext) -> TargetDecision:
        n = len(ctx.universe)
        return TargetDecision(
            weights={t: 1.0 / n for t in ctx.universe},
            reason="Rebalance to equal weights.",
        )
