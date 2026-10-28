"""Random proposals through the same risk layer, simulator and costs."""

from __future__ import annotations

import hashlib
import math

import numpy as np

from baselines.base import StrategyContext, TargetDecision


class RandomAgent:
    name = "random_agent"

    def __init__(
        self,
        seed: int = 0,
        *,
        n_names: int = 5,
        min_gross: float = 0.5,
        max_gross: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.n_names = int(n_names)
        self.min_gross = float(min_gross)
        self.max_gross = float(max_gross)

    def _rng(self, as_of) -> np.random.Generator:
        digest = hashlib.sha256(f"{self.seed}:{as_of.isoformat()}".encode()).hexdigest()
        return np.random.default_rng(int(digest[:16], 16))

    def propose(self, ctx: StrategyContext) -> TargetDecision:
        rng = self._rng(ctx.as_of)
        n = min(self.n_names, len(ctx.universe))
        chosen = list(rng.choice(list(ctx.universe), size=n, replace=False))
        raw = rng.dirichlet(np.ones(n))
        gross = float(rng.uniform(self.min_gross, self.max_gross))
        weights = {t: 0.0 for t in ctx.universe}
        for ticker, w in zip(chosen, raw):
            weights[ticker] = float(w * gross)
        return TargetDecision(
            weights=weights,
            reason="Seeded random weights, passed through the hard risk layer.",
            flags=(f"seed:{self.seed}",),
        )
