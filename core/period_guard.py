"""Period enforcement.

Every backtest declares exactly one period. The guard rejects dates outside it.
clean_test and forward additionally require --final plus a verified freeze lock.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from core.freeze import verify_freeze_lock
from core.periods import Period, PeriodTable, load_period_table


class PeriodError(ValueError):
    pass


class PeriodGuard:
    def __init__(
        self,
        table: PeriodTable,
        period_name: str,
        *,
        final: bool = False,
        freeze_path: str | Path | None = None,
        allow_forward: bool = False,
    ) -> None:
        self.table = table
        if period_name not in table.periods:
            raise PeriodError(f"Unknown period {period_name!r}")
        self.period_name = period_name
        self.period: Period = table.get(period_name)
        self.final = final
        self.allow_forward = allow_forward
        self.freeze_path = freeze_path

        if self.period.name == "warmup":
            raise PeriodError("warmup is indicator history only; no decisions allowed")
        if self.period.name == "embargo":
            raise PeriodError("embargo is excluded from all claims")

        if self.period.name in {"clean_test", "forward"}:
            if not final:
                raise PeriodError(
                    f"{self.period.name} requires --final and a verified freeze lock"
                )
            if not table.cutoff_verified:
                raise PeriodError(
                    "clean_test/forward require a human-verified model cutoff in models.yaml"
                )
            if self.period.name == "forward" and not allow_forward:
                raise PeriodError("forward paper trading requires allow_forward=True")
            ok, reason = verify_freeze_lock(self.freeze_path)
            if not ok:
                raise PeriodError(f"freeze verification failed: {reason}")

        if self.period.name == "clean_test":
            if table.clean_test_weeks < 52:
                raise PeriodError(
                    "clean_test has fewer than 52 weekly decisions; project is underpowered"
                )
            expected_min = date.fromisoformat(table.periods["embargo"].end.isoformat())
            if self.period.start <= expected_min:
                raise PeriodError(
                    "clean_test starts inside or before the cutoff+embargo window"
                )

    def validate(self, as_of: date) -> Period:
        if not self.period.contains(as_of):
            raise PeriodError(
                f"{as_of.isoformat()} is outside declared period {self.period_name} "
                f"({self.period.start.isoformat()} to "
                f"{self.period.end.isoformat() if self.period.end else 'open'})"
            )
        return self.period

    def __repr__(self) -> str:
        return f"PeriodGuard(period={self.period_name!r}, final={self.final})"


def default_guard(
    period_name: str,
    *,
    final: bool = False,
    periods_path: str | Path | None = None,
    freeze_path: str | Path | None = None,
    allow_forward: bool = False,
) -> PeriodGuard:
    table = load_period_table(periods_path)
    return PeriodGuard(
        table,
        period_name,
        final=final,
        freeze_path=freeze_path,
        allow_forward=allow_forward,
    )
