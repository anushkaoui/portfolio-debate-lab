# Multi-agent portfolio debate: one-page note

**Status:** research prototype. No performance claim is available yet. The
model cutoff is unverified and all reported development runs use a deterministic
mock model. Do not interpret any result as evidence of investment skill.

## What it does

The system gives two analyst agents (trend and contrarian), a risk-manager
agent, an optional rebuttal round and a trader a numeric, anonymized market
context ending at a Friday close. Agents emit structured proposals, critiques
and target weights. Code, not the agents, enforces the risk limits, executes at
the next trading-day open, charges costs and records the result.

## Protections built in

- One as-of time gate; no agent-visible feature after the decision close.
- Ticker/name/date anonymization in the primary context, plus identified-mode
  runs only for the contamination gap.
- Hard risk layer with single-name, volatility-scaled, group, gross, turnover,
  ADV-buy, staleness and drawdown controls; property-tested and idempotent.
- Next-open simulator with commissions, slippage, market impact and accounting
  identities.
- Clean-window design: the development period is contaminated and never used
  for claims; `embargo` is excluded; `clean_test` is a single frozen run.
- Random-agent null through the same risk layer and simulator, block bootstrap,
  deflated Sharpe, power analysis and a complete append-only trial ledger.
- Pre-registered probes, transcript leakage scan and contamination-gap test.
- `freeze.lock` hashes prompts, configs, `core/`, `agents/`, `baselines/` and
  `eval/`; the human-only final runner refuses unapproved or changed inputs.

## Current development findings (mock only; not evidence)

On the 2022-07 to 2023-12 development window the code baselines are competitive
with the mock debate system, and the mock system is below buy-and-hold on
Sharpe. The mock tuning grid also failed the walk-forward fold gate, so no
configuration is approved for freezing. This is useful as a debugging result,
not as a hypothesis test. The simulated LLM calls have no API cost and are not
representative of a live model.

## Before any final claim

A human must pin and verify the served model checkpoint and cutoff, regenerate
the period table, review dev and leakage evidence, approve the freeze, supply
sealed data and run `eval/run_final.py` exactly once. The final report must
report the pre-registered criteria, power/MDE, null percentile, DSR, robustness,
contamination evidence and the substantial small-sample and execution limits.
