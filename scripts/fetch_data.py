#!/usr/bin/env python3
"""Download OHLCV data into data/dev (builder-safe) or data/sealed (human only).

The dev scope stops at the end of precutoff_holdout. The sealed scope contains
embargo, clean_test and forward data and must only be downloaded by the human.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import load_universe  # noqa: E402


def _download_one(ticker: str, start: date, end: date) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    df = df.rename(columns=lambda c: str(c).strip().title())
    needed = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    if not needed.issubset(df.columns):
        raise RuntimeError(f"{ticker}: missing columns {sorted(needed - set(df.columns))}")
    df = df.dropna(subset=["Close"])
    factor = (df["Adj Close"] / df["Close"]).fillna(1.0)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df.index).date,
            "ticker": ticker,
            "open": df["Open"].to_numpy() * factor.to_numpy(),
            "high": df["High"].to_numpy() * factor.to_numpy(),
            "low": df["Low"].to_numpy() * factor.to_numpy(),
            "close": df["Close"].to_numpy() * factor.to_numpy(),
            "volume": df["Volume"].to_numpy(),
            "adj_close": df["Adj Close"].to_numpy(),
            "raw_close": df["Close"].to_numpy(),
        }
    )
    out["dollar_volume"] = out["raw_close"] * out["volume"]
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    return out.dropna(subset=["open", "close", "volume"]).reset_index(drop=True)


def fetch(scope: str, start: date, end: date, *, force: bool = False) -> Path:
    if scope == "dev":
        out_dir = ROOT / "data" / "dev"
    elif scope == "sealed":
        out_dir = ROOT / "data" / "sealed"
    else:
        raise ValueError("scope must be dev or sealed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prices.parquet"
    if out_path.exists() and not force:
        print(f"{out_path} exists; use --force to redownload")
        return out_path

    universe = load_universe()
    tickers = universe["tickers"]
    frames = []
    failures: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            df = _download_one(ticker, start, end)
            if df.empty:
                raise RuntimeError("empty after filtering")
            frames.append(df)
            print(f"[{i:02d}/{len(tickers)}] {ticker}: {len(df)} rows")
        except Exception as exc:
            failures.append(f"{ticker}: {exc}")
            print(f"[{i:02d}/{len(tickers)}] {ticker}: FAILED {exc}")
    if failures:
        raise RuntimeError("Download failures:\n" + "\n".join(failures))

    prices = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    # Coverage sanity check: each ticker should have at least 90% of the median
    # number of rows, so a silently truncated series cannot slip through.
    counts = prices.groupby("ticker")["date"].nunique()
    if len(counts) != len(tickers) or counts.min() < 0.9 * counts.median():
        raise RuntimeError(f"Bad ticker coverage: {counts.to_dict()}")
    prices.to_parquet(out_path, index=False)

    metadata = {
        "scope": scope,
        "source": "Yahoo Finance via yfinance",
        "terms_note": (
            "Confirm the data source's terms of use before redistributing. "
            "Adjusted OHLC uses the Adj Close/Close ratio; returns are total-return "
            "adjusted. Raw close is retained for dollar-volume approximation."
        ),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "tickers": tickers,
        "rows": int(len(prices)),
        "first_date": str(prices["date"].min()),
        "last_date": str(prices["date"].max()),
        "per_ticker_rows": {str(k): int(v) for k, v in counts.items()},
        "survivorship_note": universe.get("survivorship_note"),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path} ({len(prices)} rows, {prices['date'].min()} to {prices['date'].max()})")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["dev", "sealed"], default="dev")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--i-am-the-human",
        action="store_true",
        help="required for sealed downloads; the builder must never run this",
    )
    args = parser.parse_args(argv)
    if args.scope == "sealed" and not args.i_am_the_human:
        parser.error("sealed downloads require --i-am-the-human (human-only step)")

    if args.start:
        start = date.fromisoformat(args.start)
    else:
        # Both scopes start in 2019 so the final clean run has enough history
        # for its 252-day features. The sealed file is still human-only.
        start = date(2019, 1, 1)
    if args.end:
        end = date.fromisoformat(args.end)
    else:
        end = date(2024, 6, 28) if args.scope == "dev" else date.today()
    return 0 if fetch(args.scope, start, end, force=args.force) else 1


if __name__ == "__main__":
    raise SystemExit(main())
