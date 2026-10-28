# Portfolio Debate Lab

Research-only testbed for a multi-agent LLM portfolio debate. The system asks
two analysts, a risk manager, a rebuttal step and a trader to propose weights,
then **code disposes**: a deterministic risk layer, a next-open simulator and a
cost model decide what actually happens.

This project is not investment advice, does not connect to a brokerage, and
must not be presented as evidence of profitability from development-period
results. The only period that can support a performance claim is `clean_test`,
run once after a human-approved freeze.

## What is implemented

- Strict time gate: every agent-visible feature is computed from data at or
  before the Friday decision close. Exceptions are tested.
- Deterministic anonymization: agents see `S01...`, group codes and relative
  time labels; the human-held mapping never enters primary context.
- Pydantic schemas for every agent message, a single cached LLM path, call
  budgets, retries and safe fallbacks.
- Debate state machine: trend analyst, contrarian analyst, risk manager,
  rebuttal, trader; bounded calls per decision.
- Hard risk layer: single-name, volatility-scaled, group, gross, turnover,
  ADV-buy, staleness and drawdown rules. It accepts numbers only.
- Execution simulator: Friday close decision, next trading-day open fill,
  sells before buys, commissions, slippage, impact, fractional shares.
- Baselines: equal-weight buy-and-hold, equal-weight rebalanced, momentum,
  low-volatility, random agent, single LLM and ensemble.
- Leakage and luck controls: trace database, append-only trial ledger,
  block-bootstrap intervals, random-agent null, deflated Sharpe, power analysis,
  memorization probes, contamination-gap test and transcript scanner.
- Streamlit viewer, evaluation scripts, tests and a human-only final runner.

## Quick start

```bash
cd /Users/anushkagupta/inv_agent
.venv/bin/python -m pytest
.venv/bin/python core/periods.py --write          # roll the declared windows forward
.venv/bin/python scripts/fetch_data.py --scope dev
```

Run a development backtest with the deterministic mock model:

```bash
.venv/bin/python eval/run_backtest.py --strategy debate --period dev --mode mock
.venv/bin/python eval/run_baselines.py --period dev
.venv/bin/python eval/null_dist.py --period dev --runs 100 --workers 8
.venv/bin/python eval/power.py --period dev
.venv/bin/python eval/analyze.py --period dev
.venv/bin/python eval/robustness.py --strategy debate --period dev --mode mock
.venv/bin/python eval/probes.py --periods stress_holdout,dev,precutoff_holdout --n 10
.venv/bin/python eval/leakage_scan.py
.venv/bin/python eval/contamination_gap.py --strategy debate --period dev --mode mock
.venv/bin/python viewer/app.py                     # streamlit run viewer/app.py
```

`eval/robustness.py` also runs a matching equal-weight benchmark for each
variant. `eval/analyze.py` selects one canonical run per strategy, reports
block-bootstrap intervals for each metric, pairwise Sharpe differences and the
pre-registered criteria (only on `clean_test`).

Live agent calls require a pinned model and key:

```bash
export DEEPSEEK_API_KEY=...
.venv/bin/python eval/run_backtest.py --strategy debate --period dev --mode live
```

Live mode is refused without the key. The model cutoff is intentionally
unverified in this repository; see `config/models.yaml`.

## Human checkpoints

### M0: verify the model cutoff

1. Pin a served model snapshot (record it in `agent_model.pinned_snapshot`).
2. Find the model card / training-data disclosure, record
   `documented_cutoff`, `cutoff_source_url`, `checked_on`, set `verified: true`.
3. Regenerate `config/periods.yaml`: `.venv/bin/python core/periods.py --write`.
4. Confirm `provisional: false` and the expected clean-test dates.

The builder must not guess this value. Until it is verified, every period table
is provisional and final evaluation is blocked. The repository currently has a
draft `freeze.lock`; regenerating `config/periods.yaml` or editing any hashed
file invalidates it, so create the final freeze after all such changes.

### M9: tuning approval

Run the small pre-registered grid on `dev` only. Use `--mode mock` first to
check the plumbing, then repeat in live mode if the human API budget allows:

```bash
.venv/bin/python eval/tune.py --strategy debate --mode mock --sensitivity
.venv/bin/python eval/tune.py --strategy debate --mode live --sensitivity
```

Review fold stability, simplicity preference and sensitivity. Approve or reject
the selected configuration, then create a draft freeze. A draft is not an
approved final freeze; the builder should create it only after all `eval/`
edits are final:

```bash
.venv/bin/python core/freeze.py            # draft; not approved
.venv/bin/python core/freeze.py --verify   # after approval
```

### M10: final evaluation (human only)

```bash
# Human supplies sealed data. Never run this as the builder.
.venv/bin/python scripts/fetch_data.py --scope sealed --i-am-the-human

# After reviewing dev results and approving the freeze:
.venv/bin/python core/freeze.py --approve --note "reviewed dev results"

# Run once. A second run needs --declared-rerun before seeing results.
.venv/bin/python eval/run_final.py \
  --i-am-the-human \
  --strategy debate \
  --data data/sealed/prices.parquet \
  --mode live
```

The final runner verifies the freeze, cutoff, periods and data path, records the
run in `data/final_run_registry.jsonl`, runs the 1,000-seed random null, and
writes `data/results/final_auto_report.{json,md}`. A human must complete the
interpretation, including post-cutoff novelty checks, feasibility and alpha
decay after the freeze. Do not rerun to improve the result.

After the final system run, produce the comparison artifacts that use no new
LLM decisions. These are part of the pre-registered evidence, not extra tuning:

```bash
.venv/bin/python eval/run_baselines.py --period clean_test --final --no-trace
.venv/bin/python eval/robustness.py --strategy debate --period clean_test \
  --mode live --final
.venv/bin/python eval/probes.py --periods clean_test --data data/sealed/prices.parquet \
  --mode live --final
.venv/bin/python eval/contamination_gap.py --strategy debate --period precutoff_holdout \
  --mode live --data data/sealed/prices.parquet
.venv/bin/python eval/leakage_scan.py
.venv/bin/python eval/analyze.py --period clean_test
```

`eval/analyze.py` then writes the criteria table and `claims_allowed` flag. A
second clean-test exposure of the LLM system requires `--declared-rerun` and is
recorded in the registry.

## Data and scope

- `data/dev/prices.parquet`: builder-safe OHLCV for 2019-01-01 through
  2024-06-28 (20 US large caps).
- `data/sealed/`: human-only data. The builder must never read it.
- `config/periods.yaml`: generated windows. Warmup is indicator history only;
  embargo is excluded from claims; dev is for tuning and never for claims.
- `ledger.jsonl`: every evaluated configuration, including baselines, tuning
  variants and null configurations. Report the trial count.

## Layout

```text
agents/      schemas, prompts, LLM path, analysts, risk manager, trader, debate
baselines/   frozen simple strategies
core/        config, periods, time gate, context, anonymization, risk, simulator,
             metrics, trace, ledger, freeze, backtest runner
eval/        run, null, analyze, power, probes, leakage, gap, tuning, final
config/      frozen yardstick inputs and prompts
scripts/     data downloader
tests/       pytest suite
viewer/      Streamlit analysis UI
```

See `docs/human_checklist.md` for the sign-off list, `docs/one_page_note.md`
for the short design note and `docs/writeup.md` for the human-owned final report
skeleton.
