# Writeup

This file is the human-owned final report skeleton. The builder must not add
clean-test numbers or interpret them. Complete it only after the single approved
`clean_test` run.

## 1. Experiment design

- Model checkpoint and documented cutoff (source URL, checked-on date):
- Served checkpoint at final-run time:
- Period table and embargo:
- Universe, data source, costs:
- Baselines and ablations:

## 2. Power and minimum detectable effect

- Clean-test weekly decisions:
- Power-analysis MDE in Sharpe:
- Implications for interpretation:

## 3. Main clean-test result

- Net Sharpe, annualized return, volatility, Sortino, max drawdown, turnover,
  cost drag, hit rate, conviction calibration and block-bootstrap CIs:
- Random-agent null percentile:
- Deflated Sharpe probability and ledger trial count:
- Comparison with equal-weight and the best pre-registered simple signal:

## 4. Contamination evidence

- Contamination gap by contaminated period, with CI and threshold:
- Anonymized memorization probe rates versus chance:
- Clean-period probe rates:
- Transcript leakage flags and explanations:

## 5. Ablations and robustness

- No debate, one analyst, no risk-manager, random-proposal comparisons:
- 2x costs:
- Wednesday rebalance:
- Random half universe:
- Bi-weekly decisions:

## 6. Debate case studies

- Where the risk manager prevented an inappropriate trade:
- Where it blocked a good trade:
- Evidence of sycophancy, repeated errors or anchoring:

## 7. Pre-registered conclusion

State explicitly whether all five criteria in `config/experiment.yaml` passed.
Both "evidence of value" and "no reliable evidence of value" are acceptable
outcomes.

## 8. Limits

- Small sample, one market and one model:
- Simulated execution, capacity and costs:
- Residual leakage risk and unverified cutoff:
- No news or alternative data:
- Research only; not investment advice.

## 9. Reproduction notes

- Freeze SHA-256 and git commit:
- Command used:
- Declared rerun reason, if any:
