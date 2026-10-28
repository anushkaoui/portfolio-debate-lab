"""Execution simulator: target weights -> next-open fills -> portfolio state.

Timing is strict: a decision made at the close of `decision_date` can only fill
at the open of a strictly later trading day. Costs are charged per side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

from core.timegate import ExecutionPrices

EPS = 1e-9


class SimulationError(ValueError):
    pass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 2.0
    slippage_bps: float = 5.0
    cost_multiplier: float = 1.0
    participation_limit: float = 0.10
    impact_bps_at_full_participation: float = 10.0

    @classmethod
    def from_config(cls, cfg: Mapping) -> "CostModel":
        return cls(
            commission_bps=float(cfg.get("commission_bps", 2.0)),
            slippage_bps=float(cfg.get("slippage_bps", 5.0)),
            cost_multiplier=float(cfg.get("cost_multiplier", 1.0)),
            participation_limit=float(cfg.get("participation_limit", 0.10)),
            impact_bps_at_full_participation=float(
                cfg.get("impact_bps_at_full_participation", 10.0)
            ),
        )

    def rate(self, participation: float = 0.0) -> float:
        base = (self.commission_bps + self.slippage_bps) / 10_000.0
        impact = 0.0
        if self.participation_limit > 0 and participation > self.participation_limit:
            impact = (
                self.impact_bps_at_full_participation
                / 10_000.0
                * (participation / self.participation_limit - 1.0)
            )
        return self.cost_multiplier * (base + impact)


@dataclass
class Fill:
    ticker: str
    side: str
    shares: float
    price: float
    notional: float
    cost: float
    decision_date: date
    fill_date: date

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "side": self.side,
            "shares": round(self.shares, 6),
            "price": round(self.price, 6),
            "notional": round(self.notional, 6),
            "cost": round(self.cost, 6),
            "decision_date": self.decision_date.isoformat(),
            "fill_date": self.fill_date.isoformat(),
        }


@dataclass
class PortfolioState:
    as_of: date
    cash: float
    positions: dict[str, float]
    prices: dict[str, float]
    equity: float
    weights: dict[str, float]
    cumulative_costs: float

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "cash": round(self.cash, 6),
            "positions": {k: round(v, 8) for k, v in self.positions.items() if abs(v) > EPS},
            "equity": round(self.equity, 6),
            "weights": {k: round(v, 8) for k, v in self.weights.items() if abs(v) > EPS},
            "cumulative_costs": round(self.cumulative_costs, 6),
        }


@dataclass
class RebalanceResult:
    state: PortfolioState
    fills: list[Fill] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    turnover: float = 0.0
    costs: float = 0.0
    target_weights: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fills": [f.to_dict() for f in self.fills],
            "skipped": self.skipped,
            "turnover": round(self.turnover, 8),
            "costs": round(self.costs, 6),
            "target_weights": {k: round(v, 8) for k, v in self.target_weights.items()},
            "state": self.state.to_dict(),
        }


class PortfolioSimulator:
    def __init__(
        self,
        prices: ExecutionPrices,
        universe: list[str],
        cost_model: CostModel,
        *,
        initial_capital: float = 100_000.0,
        max_order_fraction_of_adv: float = 0.10,
    ) -> None:
        self.prices = prices
        self.universe = sorted(set(universe))
        self.cost_model = cost_model
        self.initial_capital = float(initial_capital)
        self.max_order_fraction_of_adv = float(max_order_fraction_of_adv)
        self.cash = float(initial_capital)
        self.positions: dict[str, float] = {t: 0.0 for t in self.universe}
        self.cumulative_costs = 0.0
        self.history: list[PortfolioState] = []
        self.last_state: PortfolioState | None = None

    # ---- valuation -----------------------------------------------------
    def _last_close(self, ticker: str, day: date) -> float | None:
        return self.prices.last_close(ticker, day)

    def mark_to_market(self, day: date) -> PortfolioState:
        prices: dict[str, float] = {}
        value = 0.0
        for ticker, shares in self.positions.items():
            price = self._last_close(ticker, day)
            if price is None:
                continue
            prices[ticker] = float(price)
            value += shares * float(price)
        equity = self.cash + value
        weights = {
            t: (shares * prices.get(t, 0.0) / equity if equity > 0 else 0.0)
            for t, shares in self.positions.items()
        }
        state = PortfolioState(
            as_of=day,
            cash=self.cash,
            positions=dict(self.positions),
            prices=prices,
            equity=equity,
            weights=weights,
            cumulative_costs=self.cumulative_costs,
        )
        self._check_identity(state)
        return state

    def _check_identity(self, state: PortfolioState) -> None:
        value = sum(
            shares * state.prices.get(ticker, 0.0)
            for ticker, shares in state.positions.items()
        )
        if abs((state.cash + value) - state.equity) > max(1e-6, 1e-9 * abs(state.equity)):
            raise SimulationError("Accounting identity violated: equity != cash + positions")
        if state.cash < -1e-7:
            raise SimulationError(f"Negative cash: {state.cash}")

    def current_weights(self, day: date) -> dict[str, float]:
        return self.mark_to_market(day).weights

    # ---- trading -------------------------------------------------------
    def rebalance(
        self,
        target_weights: Mapping[str, float],
        decision_date: date,
        *,
        fill_date: date | None = None,
    ) -> RebalanceResult:
        if fill_date is None:
            fill_date = self.prices.next_trading_day(decision_date)
        if fill_date <= decision_date:
            raise SimulationError("Fills must occur strictly after the decision date")
        if not self.prices.has_date(decision_date) or not self.prices.has_date(fill_date):
            raise SimulationError("decision or fill date is not a trading day in the data")

        targets: dict[str, float] = {}
        for ticker, weight in target_weights.items():
            t = str(ticker)
            if t not in self.universe:
                raise SimulationError(f"Unknown ticker in target weights: {t}")
            w = float(weight)
            if w < -EPS or w > 1.0 + EPS:
                raise SimulationError(f"Target weight out of bounds for {t}: {w}")
            targets[t] = max(0.0, w)
        for ticker in self.universe:
            targets.setdefault(ticker, 0.0)
        if sum(targets.values()) > 1.0 + 1e-6:
            raise SimulationError("Target weights sum to more than 1.0 (leverage)")

        decision_state = self.mark_to_market(decision_date)
        equity_for_targets = decision_state.equity
        target_dollars = {t: w * equity_for_targets for t, w in targets.items()}

        fills: list[Fill] = []
        skipped: list[str] = []
        costs = 0.0
        traded_notional = 0.0

        buys: list[tuple[str, float, float]] = []  # ticker, notional, rate
        # Sells first.
        for ticker in self.universe:
            open_price = self.prices.open(ticker, fill_date)
            if open_price is None or open_price <= 0:
                if abs(targets[ticker] - decision_state.weights.get(ticker, 0.0)) > 1e-8:
                    skipped.append(ticker)
                continue
            current_shares = self.positions.get(ticker, 0.0)
            current_value = current_shares * open_price
            delta = target_dollars[ticker] - current_value
            if delta < -EPS and current_shares > 0:
                sell_shares = min(current_shares, -delta / open_price)
                notional = sell_shares * open_price
                adv = self.prices.adv_notional(ticker, decision_date) or 0.0
                participation = notional / adv if adv > 0 else 1.0
                rate = self.cost_model.rate(participation)
                cost = notional * rate
                self.cash += notional - cost
                self.positions[ticker] -= sell_shares
                self.cumulative_costs += cost
                costs += cost
                traded_notional += notional
                if self.positions[ticker] < EPS:
                    self.positions[ticker] = 0.0
                fills.append(
                    Fill(ticker, "sell", sell_shares, open_price, notional, cost, decision_date, fill_date)
                )
                delta += notional
            if delta > EPS:
                adv = self.prices.adv_notional(ticker, decision_date) or 0.0
                # The risk layer should have capped this; cap again defensively.
                if adv > 0:
                    max_notional = self.max_order_fraction_of_adv * adv
                    delta = min(delta, max_notional)
                participation = delta / adv if adv > 0 else 1.0
                buys.append((ticker, delta, self.cost_model.rate(participation)))

        total_buy_cost = sum(n * (1.0 + r) for _, n, r in buys)
        scale = 1.0
        if total_buy_cost > self.cash + EPS:
            scale = max(0.0, self.cash / total_buy_cost) if total_buy_cost > 0 else 0.0
        for ticker, notional0, rate in buys:
            notional = notional0 * scale
            if notional <= EPS:
                continue
            open_price = self.prices.open(ticker, fill_date)
            if open_price is None or open_price <= 0:
                skipped.append(ticker)
                continue
            shares = notional / open_price
            total = notional * (1.0 + rate)
            if total > self.cash + 1e-6:
                notional = self.cash / (1.0 + rate)
                total = notional * (1.0 + rate)
                shares = notional / open_price
            if notional <= EPS:
                continue
            cost = notional * rate
            self.cash -= total
            self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
            self.cumulative_costs += cost
            costs += cost
            traded_notional += notional
            fills.append(
                Fill(ticker, "buy", shares, open_price, notional, cost, decision_date, fill_date)
            )

        if self.cash < -1e-6:
            raise SimulationError(f"Negative cash after fills: {self.cash}")
        self.cash = max(0.0, self.cash)
        state = self.mark_to_market(fill_date)
        turnover = 0.5 * traded_notional / equity_for_targets if equity_for_targets > 0 else 0.0
        self.last_state = state
        self.history.append(state)
        return RebalanceResult(
            state=state,
            fills=fills,
            skipped=sorted(set(skipped)),
            turnover=float(turnover),
            costs=float(costs),
            target_weights=targets,
        )

    def record(self, day: date) -> PortfolioState:
        state = self.mark_to_market(day)
        self.last_state = state
        self.history.append(state)
        return state
