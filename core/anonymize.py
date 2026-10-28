"""Deterministic anonymization of assets, groups and time.

Mapping files live outside the agent context. Agents see S01... and G01...
codes and relative time labels only.
"""

from __future__ import annotations

import json
import random
import re
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.config import ROOT, stable_json

DATE_PATTERNS = [
    re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"),
]
MONTH_OR_DAY = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|"
    r"Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    re.IGNORECASE,
)


class AnonymizationLeak(RuntimeError):
    pass


class Anonymizer:
    def __init__(
        self,
        tickers: Iterable[str],
        groups: Mapping[str, str] | None = None,
        names: Mapping[str, str] | None = None,
        *,
        seed: int = 20261028,
        mapping_path: str | Path | None = None,
    ) -> None:
        self.tickers = sorted({str(t).upper() for t in tickers})
        self.names = {str(k).upper(): str(v) for k, v in (names or {}).items()}
        self.seed = int(seed)
        self.mapping_path = Path(mapping_path) if mapping_path else None

        shuffled = list(self.tickers)
        random.Random(self.seed).shuffle(shuffled)
        width = max(2, len(str(len(shuffled))))
        self._to_code = {t: f"S{i + 1:0{width}d}" for i, t in enumerate(shuffled)}
        self._to_ticker = {v: k for k, v in self._to_code.items()}

        raw_groups = {str(t): str(groups.get(t, f"GROUP_{t}")) for t in self.tickers} if groups else {}
        unique_groups = sorted(set(raw_groups.values()))
        gwidth = max(2, len(str(len(unique_groups))))
        self._group_to_code = {g: f"G{i + 1:0{gwidth}d}" for i, g in enumerate(unique_groups)}
        self._ticker_group_code = {t: self._group_to_code[g] for t, g in raw_groups.items()}
        self._ticker_group_name = {t: g for t, g in raw_groups.items()}

    # ---- asset mapping -------------------------------------------------
    def code(self, ticker: str) -> str:
        key = str(ticker).upper()
        if key not in self._to_code:
            raise KeyError(f"Ticker {ticker!r} is not in the anonymized universe")
        return self._to_code[key]

    def ticker(self, code: str) -> str:
        if code not in self._to_ticker:
            raise KeyError(f"Unknown anonymous code {code!r}")
        return self._to_ticker[code]

    @property
    def codes(self) -> list[str]:
        return sorted(self._to_ticker)

    def code_to_ticker(self) -> dict[str, str]:
        return dict(self._to_ticker)

    def ticker_to_code(self) -> dict[str, str]:
        return dict(self._to_code)

    def group_code(self, ticker: str) -> str:
        key = str(ticker).upper()
        return self._ticker_group_code.get(key, "G00")

    def group_name(self, ticker: str) -> str:
        return self._ticker_group_name.get(str(ticker).upper(), "Unknown")

    # ---- time mapping --------------------------------------------------
    @staticmethod
    def relative_time(as_of: date, observation: date | None = None) -> str:
        if observation is None or observation == as_of:
            return "t-0"
        days = (as_of - observation).days
        if days <= 0:
            return "t-0"
        if days % 7 == 0:
            return f"t-{days // 7}w"
        return f"t-{days}d"

    def payload(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "ticker_to_code": dict(self._to_code),
            "code_to_ticker": dict(self._to_ticker),
            "group_map": dict(self._ticker_group_code),
            "group_to_code": dict(self._group_to_code),
            "group_names": dict(self._ticker_group_name),
            "names": dict(self.names),
            "note": "Human-held mapping. Never include this file in agent context.",
        }

    def save(self) -> Path | None:
        if not self.mapping_path:
            return None
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.write_text(json.dumps(self.payload(), indent=2), encoding="utf-8")
        return self.mapping_path

    @classmethod
    def load_or_create(
        cls,
        tickers: Iterable[str],
        groups: Mapping[str, str],
        names: Mapping[str, str] | None = None,
        *,
        seed: int = 20261028,
        mapping_path: str | Path | None = None,
    ) -> "Anonymizer":
        path = Path(mapping_path) if mapping_path else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            mapping = {str(k).upper(): str(v) for k, v in data["ticker_to_code"].items()}
            obj = cls.__new__(cls)
            obj.tickers = sorted(mapping)
            obj.names = {str(k).upper(): str(v) for k, v in (data.get("names") or names or {}).items()}
            obj.seed = int(data.get("seed", seed))
            obj.mapping_path = path
            obj._to_code = mapping
            obj._to_ticker = {v: k for k, v in mapping.items()}
            obj._ticker_group_code = {
                str(k).upper(): str(v) for k, v in (data.get("group_map") or {}).items()
            }
            obj._group_to_code = {
                str(k): str(v) for k, v in (data.get("group_to_code") or {}).items()
            }
            obj._ticker_group_name = {
                str(k).upper(): str(v) for k, v in (data.get("group_names") or {}).items()
            }
            if set(obj.tickers) != {str(t).upper() for t in tickers}:
                raise AnonymizationLeak(
                    "Existing anonymization map does not match the configured universe"
                )
            return obj
        obj = cls(
            tickers,
            groups,
            names,
            seed=seed,
            mapping_path=path,
        )
        obj.save()
        return obj

    # ---- leakage checks ------------------------------------------------
    def find_identifiers(self, text: str) -> list[str]:
        findings: list[str] = []
        for ticker in self.tickers:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", text):
                findings.append(f"ticker:{ticker}")
        for name in self.names.values():
            if name and name.lower() in text.lower():
                findings.append(f"name:{name}")
        for pattern in DATE_PATTERNS:
            for match in pattern.findall(text):
                findings.append(f"date:{match}")
        for match in MONTH_OR_DAY.findall(text):
            findings.append(f"calendar_word:{match}")
        if re.search(r"(?<![\d.])(?:Q[1-4]|quarter\s+[1-4]|20\d{2})(?![\d.])", text):
            findings.append("calendar_period")
        return sorted(set(findings))

    def assert_clean(self, context: Any) -> None:
        text = context if isinstance(context, str) else stable_json(context)
        findings = self.find_identifiers(text)
        if findings:
            raise AnonymizationLeak(
                "Agent context contains identifiers: " + ", ".join(findings[:20])
            )


def build_anonymizer_for_universe(
    universe: Mapping[str, Any],
    *,
    seed: int = 20261028,
    mapping_path: str | Path | None = None,
) -> Anonymizer:
    path = mapping_path or (ROOT / "data" / "dev" / "anonymization_map.json")
    group_map = {str(k).upper(): str(v) for k, v in (universe.get("group_map") or {}).items()}
    obj = Anonymizer.load_or_create(
        universe["tickers"],
        group_map,
        universe.get("companies", {}),
        seed=seed,
        mapping_path=path,
    )
    # Older mapping files may predate group names. Reconcile from config without
    # changing the human-held ticker/code mapping.
    obj._ticker_group_name = {t: group_map.get(t, "Unknown") for t in obj.tickers}
    missing_codes = {g for g in group_map.values() if g not in obj._group_to_code}
    if missing_codes:
        for g in sorted(missing_codes):
            obj._group_to_code[g] = f"G{len(obj._group_to_code) + 1:02d}"
    obj._ticker_group_code = {
        t: obj._group_to_code.get(g, "G00") for t, g in group_map.items()
    }
    return obj
