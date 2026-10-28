#!/usr/bin/env python3
"""Streamlit decision and results viewer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.trace import TraceStore  # noqa: E402

RESULT_DIR = ROOT / "data" / "results"
TRACE_DB = ROOT / "data" / "traces.sqlite"
MAPPING_PATH = ROOT / "data" / "dev" / "anonymization_map.json"

st.set_page_config(page_title="Portfolio debate lab", layout="wide")
st.title("Portfolio debate lab")
st.caption("Research view. Codes are anonymous unless you explicitly reveal the human-held mapping.")


def _loads(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


@st.cache_data(show_spinner=False)
def load_results() -> list[dict]:
    rows = []
    for path in sorted(RESULT_DIR.glob("*.json")):
        if path.name.startswith(("analysis_", "null_", "power_", "probes_", "leakage_", "contamination_", "tuning_", "final_auto")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "run_id" in data and "metrics" in data:
            rows.append(data)
    return rows


@st.cache_data(show_spinner=False)
def load_mapping() -> dict:
    if MAPPING_PATH.exists():
        return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    return {}


def reveal(code: str, mapping: dict, enabled: bool) -> str:
    if not enabled:
        return code
    ticker = (mapping.get("code_to_ticker") or {}).get(code)
    name = (mapping.get("names") or {}).get(ticker or "", "")
    return f"{code} = {ticker} ({name})" if ticker else code


mode = st.sidebar.radio("View", ["Results", "Decision explorer", "Ledger", "Data health"])
results = load_results()
mapping = load_mapping()
show_real = st.sidebar.toggle("Reveal tickers/names (analysis only)", value=False)

if mode == "Results":
    st.subheader("Saved runs")
    if not results:
        st.info("No saved results yet. Run a dev backtest first.")
    else:
        table = []
        for r in results:
            m = r.get("metrics", {})
            table.append(
                {
                    "run_id": r["run_id"],
                    "strategy": r.get("strategy"),
                    "period": r.get("period"),
                    "mode": r.get("mode"),
                    "sharpe": m.get("sharpe"),
                    "ann_return": m.get("annualized_return"),
                    "max_dd": m.get("max_drawdown"),
                    "cost_drag": m.get("cost_drag_of_initial"),
                    "decisions": m.get("decisions"),
                    "provisional": m.get("provisional_periods"),
                }
            )
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        if any(r.get("metrics", {}).get("provisional_periods") for r in results):
            st.warning("Provisional periods: the model cutoff is unverified; treat all results as exploratory.")
        chosen = st.selectbox("Equity curve", [r["run_id"] for r in results])
        result = next(r for r in results if r["run_id"] == chosen)
        eq = pd.DataFrame(result["equity"])
        if not eq.empty:
            eq["date"] = pd.to_datetime(eq["date"])
            st.line_chart(eq.set_index("date")["equity"])
        st.json(result["metrics"])
        null_path = RESULT_DIR / f"null_{result.get('period')}.json"
        if null_path.exists():
            st.subheader("Random-agent null")
            null = json.loads(null_path.read_text(encoding="utf-8"))
            st.json(null.get("metrics"))
        gap_path = RESULT_DIR / "contamination_gap.json"
        if gap_path.exists() and result.get("period") in {"dev", "stress_holdout", "precutoff_holdout"}:
            st.subheader("Contamination gap")
            st.json(json.loads(gap_path.read_text(encoding="utf-8")))
        analysis_path = RESULT_DIR / f"analysis_{result.get('period')}.json"
        if analysis_path.exists():
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            st.caption(f"Ledger trials at analysis: {analysis.get('ledger_trials')}")

elif mode == "Decision explorer":
    if not TRACE_DB.exists():
        st.info("No trace database yet.")
    else:
        store = TraceStore(TRACE_DB)
        runs = store.runs()
        if not runs:
            st.info("No traced runs yet.")
        else:
            labels = {f"{r['run_id']} | {r.get('strategy')} | {r.get('period')}": r for r in runs}
            selected_label = st.selectbox("Run", list(labels))
            run = labels[selected_label]
            decisions = store.decisions(run["run_id"])
            st.caption(
                f"model={run.get('model')} mode={run.get('mode')} calls={run.get('calls')} "
                f"cost=${run.get('cost_usd', 0):.4f}"
            )
            if not decisions:
                st.info("No decisions in this run.")
            else:
                dates = [d["decision_date"] for d in decisions]
                chosen_date = st.select_slider("Decision date", options=dates, value=dates[0])
                row = next(d for d in decisions if d["decision_date"] == chosen_date)
                left, right = st.columns([1, 1])
                with left:
                    st.subheader("Context")
                    st.json(_loads(row.get("context_json")) or {})
                with right:
                    st.subheader("Risk verdict")
                    st.json(_loads(row.get("risk_json")) or {})
                    st.subheader("Fills")
                    st.json(_loads(row.get("fills_json")) or [])
                    st.subheader("Portfolio")
                    st.json(_loads(row.get("portfolio_json")) or {})
                st.subheader("Debate transcript")
                st.json(_loads(row.get("debate_json")) or {})
                if show_real and mapping:
                    st.subheader("Mapped context")
                    context = _loads(row.get("context_json")) or {}
                    unmasked = {}
                    for code, feats in (context.get("features") or {}).items():
                        unmasked[reveal(code, mapping, True)] = feats
                    st.json(unmasked)
                st.subheader("Warnings")
                st.write(run.get("notes") or "")

elif mode == "Ledger":
    ledger_path = ROOT / "ledger.jsonl"
    if not ledger_path.exists():
        st.info("No ledger yet.")
    else:
        rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        st.metric("Trials", len(rows))
        df = pd.DataFrame(
            [
                {
                    "trial": r.get("trial"),
                    "timestamp": r.get("timestamp_utc"),
                    "strategy": r.get("strategy"),
                    "period": r.get("period"),
                    "mode": r.get("mode"),
                    "sharpe": (r.get("metrics") or {}).get("sharpe"),
                    "cost_usd": r.get("cost_usd"),
                    "run_id": r.get("run_id"),
                }
                for r in rows
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.subheader("Data health")
    meta = ROOT / "data" / "dev" / "metadata.json"
    if meta.exists():
        st.json(json.loads(meta.read_text(encoding="utf-8")))
    else:
        st.info("No dev metadata.")
    st.write("Sealed data status is intentionally not shown. Final runs are human-only.")
    freeze = ROOT / "freeze.lock"
    if freeze.exists():
        lock = json.loads(freeze.read_text(encoding="utf-8"))
        st.write("Freeze lock:", "approved" if lock.get("approved") else "DRAFT")
        st.write("Cutoff verified:", lock.get("model_cutoff_verified"))
    else:
        st.info("No freeze lock yet.")
