"""Period design and generation from a model's documented cutoff.

The formulas are the ones in the build specification, section 4.3. Boundaries
are aligned to the NYSE trading calendar where the worked example requires it.
The generated table is written to config/periods.yaml and is part of the freeze.
"""

from __future__ import annotations

import calendar as _calendar
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd
from dateutil.relativedelta import relativedelta

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import ROOT, load_yaml, sha256_obj  # noqa: E402

PeriodName = Literal[
    "warmup",
    "stress_holdout",
    "dev",
    "precutoff_holdout",
    "embargo",
    "clean_test",
    "forward",
]

DECISION_WEEKDAY = 4  # Friday


@dataclass(frozen=True)
class Period:
    name: str
    start: date
    end: date | None
    contaminated: str
    allowed_use: str
    decisions_allowed: bool = True

    def contains(self, day: date) -> bool:
        if day < self.start:
            return False
        return self.end is None or day <= self.end

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat() if self.end else None
        return d


@dataclass(frozen=True)
class PeriodTable:
    periods: dict[str, Period]
    cutoff: date
    cutoff_verified: bool
    provisional: bool
    generated_on: date
    embargo_months: int
    clean_test_weeks: int
    underpowered: bool
    warnings: tuple[str, ...] = ()

    def get(self, name: str) -> Period:
        if name not in self.periods:
            raise KeyError(f"Unknown period {name!r}; have {sorted(self.periods)}")
        return self.periods[name]

    def __iter__(self):
        return iter(self.periods.values())

    def to_dict(self) -> dict:
        return {
            "cutoff": self.cutoff.isoformat(),
            "cutoff_verified": self.cutoff_verified,
            "provisional": self.provisional,
            "generated_on": self.generated_on.isoformat(),
            "embargo_months": self.embargo_months,
            "clean_test_weeks": self.clean_test_weeks,
            "underpowered": self.underpowered,
            "warnings": list(self.warnings),
            "periods": {name: p.to_dict() for name, p in self.periods.items()},
        }

    @property
    def digest(self) -> str:
        return sha256_obj(self.to_dict())


def _nyse_calendar():
    import exchange_calendars as xcals

    return xcals.get_calendar("XNYS")


def _as_ts(d: date) -> pd.Timestamp:
    return pd.Timestamp(d.year, d.month, d.day)


def is_trading_day(d: date) -> bool:
    return bool(_nyse_calendar().is_session(_as_ts(d)))


def first_session_on_or_after(d: date) -> date:
    cal = _nyse_calendar()
    ts = cal.date_to_session(_as_ts(d), direction="next")
    return ts.date()


def last_session_on_or_before(d: date) -> date:
    cal = _nyse_calendar()
    ts = cal.date_to_session(_as_ts(d), direction="previous")
    return ts.date()


def first_session_after(d: date) -> date:
    if is_trading_day(d):
        return last_session_on_or_before(d + timedelta(days=1))
    return first_session_on_or_after(d)


def add_months(d: date, months: int) -> date:
    return (pd.Timestamp(d) + relativedelta(months=months)).date()


def month_end(d: date) -> date:
    return date(d.year, d.month, _calendar.monthrange(d.year, d.month)[1])


def last_complete_friday(today: date | None = None) -> date:
    """Latest Friday strictly before `today`, aligned back to a trading session."""
    today = today or date.today()
    offset = (today.weekday() - DECISION_WEEKDAY) % 7
    if offset == 0:
        offset = 7
    candidate = today - timedelta(days=offset)
    return last_session_on_or_before(candidate)


def next_friday_on_or_after(d: date) -> date:
    offset = (DECISION_WEEKDAY - d.weekday()) % 7
    return d + timedelta(days=offset)


def count_weekly_decisions(start: date, end: date) -> int:
    if end < start:
        return 0
    first = next_friday_on_or_after(start)
    count = 0
    d = first
    while d <= end:
        count += 1
        d += timedelta(days=7)
    return count


def generate_periods(
    cutoff: date,
    *,
    cutoff_verified: bool = True,
    today: date | None = None,
    embargo_months: int = 6,
    stress_start: date = date(2020, 1, 1),
    min_clean_weeks: int = 52,
) -> PeriodTable:
    if embargo_months < 3:
        raise ValueError("Embargo must be at least 3 months per the specification")
    today = today or date.today()

    stress_first = first_session_on_or_after(stress_start)
    stress_last = last_session_on_or_before(add_months(cutoff, -24))
    warnings: list[str] = []
    periods: dict[str, Period] = {}

    stress_exists = (stress_last - stress_first).days >= 365
    if stress_exists:
        periods["stress_holdout"] = Period(
            name="stress_holdout",
            start=stress_first,
            end=stress_last,
            contaminated="yes_heavily",
            allowed_use="Regime robustness and contamination-gap test; never tuned on.",
        )
        first_period_start = stress_first
    else:
        warnings.append(
            "stress_holdout would be shorter than 12 months and was dropped."
        )
        first_period_start = first_session_on_or_after(add_months(cutoff, -24))

    warmup_anchor = stress_start if stress_exists else first_session_on_or_after(add_months(cutoff, -24))
    warmup_start = add_months(warmup_anchor, -12)
    warmup_end = warmup_anchor - timedelta(days=1)
    periods["warmup"] = Period(
        name="warmup",
        start=warmup_start,
        end=warmup_end,
        contaminated="n/a",
        allowed_use="Indicator history only. No decisions.",
        decisions_allowed=False,
    )

    dev_start = first_session_after(add_months(cutoff, -24))
    dev_end = last_session_on_or_before(add_months(cutoff, -6))
    periods["dev"] = Period(
        name="dev",
        start=dev_start,
        end=dev_end,
        contaminated="yes",
        allowed_use="Only period for tuning prompts, protocol, thresholds and risk limits. Anonymized. No performance claims.",
    )

    precutoff_start = first_session_on_or_after(dev_end + timedelta(days=1))
    precutoff_end = last_session_on_or_before(cutoff)
    if precutoff_end >= precutoff_start:
        periods["precutoff_holdout"] = Period(
            name="precutoff_holdout",
            start=precutoff_start,
            end=precutoff_end,
            contaminated="partially",
            allowed_use="Pre-cutoff check after the dev freeze; contamination-gap test.",
        )
    else:
        warnings.append("precutoff_holdout is empty and was dropped.")

    embargo_start = cutoff + timedelta(days=1)
    embargo_end = month_end(add_months(cutoff, embargo_months))
    periods["embargo"] = Period(
        name="embargo",
        start=embargo_start,
        end=embargo_end,
        contaminated="uncertain",
        allowed_use="Excluded from all claims. Guards against fuzzy cutoffs.",
        decisions_allowed=False,
    )

    clean_start = first_session_on_or_after(embargo_end + timedelta(days=1))
    clean_end = last_complete_friday(today)
    clean_weeks = count_weekly_decisions(clean_start, clean_end)
    underpowered = clean_weeks < min_clean_weeks
    periods["clean_test"] = Period(
        name="clean_test",
        start=clean_start,
        end=clean_end,
        contaminated="no",
        allowed_use="Run once, after freeze. The only period that supports performance claims.",
    )
    if underpowered:
        warnings.append(
            f"clean_test has {clean_weeks} weekly decisions (< {min_clean_weeks}); "
            "project is underpowered: no performance claims."
        )

    forward_start = next_friday_on_or_after(clean_end + timedelta(days=1))
    periods["forward"] = Period(
        name="forward",
        start=forward_start,
        end=None,
        contaminated="no",
        allowed_use="Live paper trading. One decision per week, logged as it happens.",
    )

    ordered = {key: periods[key] for key in [
        "warmup",
        "stress_holdout",
        "dev",
        "precutoff_holdout",
        "embargo",
        "clean_test",
        "forward",
    ] if key in periods}

    provisional = not cutoff_verified
    if provisional:
        warnings.append(
            "PROVISIONAL: the model cutoff is not human-verified. Do not run final evaluation."
        )

    return PeriodTable(
        periods=ordered,
        cutoff=cutoff,
        cutoff_verified=cutoff_verified,
        provisional=provisional,
        generated_on=today,
        embargo_months=embargo_months,
        clean_test_weeks=clean_weeks,
        underpowered=underpowered,
        warnings=tuple(warnings),
    )


def load_model_cutoff(
    models_path: str | Path | None = None,
) -> tuple[date, bool, tuple[str, ...]]:
    """Return (cutoff, verified, warnings) from config/models.yaml.

    If the human has not verified a cutoff, the documented worked-example
    placeholder is used for development only and is flagged.
    """
    data = load_yaml(models_path or ROOT / "config" / "models.yaml")
    model = data.get("agent_model", {})
    periods_cfg = data.get("periods", {})
    warnings: list[str] = []
    verified = bool(model.get("verified")) and bool(model.get("documented_cutoff"))
    if verified:
        cutoff = date.fromisoformat(str(model["documented_cutoff"]))
        if not model.get("cutoff_source_url"):
            raise ValueError("Verified models.yaml must include cutoff_source_url")
        return cutoff, True, tuple(warnings)

    raw = periods_cfg.get("cutoff_provisional")
    if not raw:
        raise ValueError(
            "No verified cutoff and no periods.cutoff_provisional in models.yaml. "
            "The builder must stop and ask the human."
        )
    cutoff = date.fromisoformat(str(raw))
    warnings.append(
        "Using periods.cutoff_provisional because agent_model.documented_cutoff "
        "is not human-verified."
    )
    return cutoff, False, tuple(warnings)


def table_from_config(
    models_path: str | Path | None = None, today: date | None = None
) -> PeriodTable:
    data = load_yaml(models_path or ROOT / "config" / "models.yaml")
    cfg = data.get("periods", {})
    cutoff, verified, warnings = load_model_cutoff(models_path)
    table = generate_periods(
        cutoff,
        cutoff_verified=verified,
        today=today,
        embargo_months=int(cfg.get("embargo_months", 6)),
        stress_start=date.fromisoformat(str(cfg.get("stress_start", "2020-01-01"))),
        min_clean_weeks=int(cfg.get("min_clean_weeks", 52)),
    )
    return PeriodTable(
        periods=table.periods,
        cutoff=table.cutoff,
        cutoff_verified=table.cutoff_verified,
        provisional=table.provisional,
        generated_on=table.generated_on,
        embargo_months=table.embargo_months,
        clean_test_weeks=table.clean_test_weeks,
        underpowered=table.underpowered,
        warnings=tuple(warnings) + table.warnings,
    )


def write_periods_yaml(
    table: PeriodTable, path: str | Path | None = None
) -> Path:
    import yaml

    path = Path(path or ROOT / "config" / "periods.yaml")
    payload = table.to_dict()
    payload["digest"] = table.digest
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    return path


def load_period_table(path: str | Path | None = None) -> PeriodTable:
    import yaml

    p = Path(path or ROOT / "config" / "periods.yaml")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    periods = {
        name: Period(
            name=name,
            start=date.fromisoformat(row["start"]),
            end=date.fromisoformat(row["end"]) if row.get("end") else None,
            contaminated=str(row.get("contaminated", "")),
            allowed_use=str(row.get("allowed_use", "")),
            decisions_allowed=bool(row.get("decisions_allowed", True)),
        )
        for name, row in data["periods"].items()
    }
    return PeriodTable(
        periods=periods,
        cutoff=date.fromisoformat(data["cutoff"]),
        cutoff_verified=bool(data.get("cutoff_verified", False)),
        provisional=bool(data.get("provisional", True)),
        generated_on=date.fromisoformat(data["generated_on"]),
        embargo_months=int(data.get("embargo_months", 6)),
        clean_test_weeks=int(data.get("clean_test_weeks", 0)),
        underpowered=bool(data.get("underpowered", True)),
        warnings=tuple(data.get("warnings") or ()),
    )


def trading_days(start: date, end: date) -> list[date]:
    cal = _nyse_calendar()
    sessions = cal.sessions_in_range(_as_ts(start), _as_ts(end))
    return [ts.date() for ts in sessions]


def weekly_decision_dates(
    start: date, end: date, frequency: str = "weekly", weekday: int = DECISION_WEEKDAY
) -> list[date]:
    """Decision dates at the close of `weekday`.

    The first decision is the first such weekday on or after `start`. If that
    weekday is a market holiday, the previous trading session is used.
    """
    if end < start:
        return []
    offset = (weekday - start.weekday()) % 7
    d = start + timedelta(days=offset)
    step = 14 if frequency == "biweekly" else 7
    out: list[date] = []
    while d <= end:
        session = last_session_on_or_before(d)
        if session >= start and session <= end and session not in out:
            out.append(session)
        d += timedelta(days=step)
    return sorted(out)


def decision_dates_for_period(
    table: PeriodTable, period_name: str, frequency: str = "weekly", weekday: int = DECISION_WEEKDAY
) -> list[date]:
    period = table.get(period_name)
    if not period.decisions_allowed or period.end is None:
        return []
    return weekly_decision_dates(period.start, period.end, frequency, weekday)


def format_table(table: PeriodTable) -> str:
    rows = [
        (
            p.name,
            p.start.isoformat(),
            p.end.isoformat() if p.end else "open-ended",
            p.contaminated,
            p.allowed_use,
        )
        for p in table
    ]
    widths = [max(len(str(r[i])) for r in [("Period", "Start", "End", "Contaminated", "Allowed use")] + rows) for i in range(5)]
    header = ("Period", "Start", "End", "Contaminated", "Allowed use")
    lines = ["  ".join(str(header[i]).ljust(widths[i]) for i in range(5))]
    lines.append("  ".join("-" * widths[i] for i in range(5)))
    for row in rows:
        lines.append("  ".join(str(row[i]).ljust(widths[i]) for i in range(5)))
    lines.append("")
    lines.append(f"model cutoff: {table.cutoff.isoformat()} (verified={table.cutoff_verified})")
    lines.append(f"weekly decisions in clean_test: {table.clean_test_weeks}; underpowered={table.underpowered}")
    for warning in table.warnings:
        lines.append(f"WARNING: {warning}")
    return "\n".join(lines)


def decisions_within(start: date, end: date, dates: Iterable[date]) -> list[date]:
    return [d for d in dates if start <= d <= end]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate config/periods.yaml from models.yaml")
    parser.add_argument("--models", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--today", default=None)
    parser.add_argument("--write", action="store_true", help="write config/periods.yaml")
    args = parser.parse_args(argv)
    today = date.fromisoformat(args.today) if args.today else None
    table = table_from_config(args.models, today=today)
    print(format_table(table))
    if args.write:
        out = write_periods_yaml(table, args.out)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
