# Human sign-off checklist

## Before any final run

- [ ] Scope acknowledged: research/learning only, no brokerage, not advice.
- [ ] Agent model checkpoint pinned exactly; `pinned_snapshot` records the served version.
- [ ] `documented_cutoff` verified from a primary source; source URL and date recorded.
- [ ] `agent_model.verified: true` in `config/models.yaml`.
- [ ] `config/periods.yaml` regenerated and `provisional: false`.
- [ ] Clean-test window still has at least 52 weekly decisions.
- [ ] Dev-only tuning completed; every trial is in `ledger.jsonl`.
- [ ] Selected configuration is the simplest one within the pre-registered tolerance.
- [ ] Sensitivity checks reviewed; knife-edge settings rejected in favor of defaults.
- [ ] Walk-forward fold gate reviewed; if `no_feasible_config` is true, do not
      freeze without an explicit human decision and a documented reason.
- [ ] Dev mock and live results reviewed as exploratory only.
- [ ] Costs, turnover, capacity and zero-cost audit reviewed.
- [ ] `freeze.lock` created and approved; every hashed file is final.
- [ ] If a draft lock exists, it was regenerated after the last hashed-file edit
      and before approval.
- [ ] Sealed data downloaded by the human into `data/sealed/`, never by the builder.
- [ ] Data terms of use reviewed; no redistribution without permission.
- [ ] Final rerun policy understood: one run; any rerun needs a declared reason
      recorded before seeing results.

## After the single clean-test run

- [ ] Final metrics, block-bootstrap CI, random-null percentile, DSR and power MDE reported.
- [ ] Ledger trial count visible; no hidden multiplicity.
- [ ] Power limitation stated: effects below the MDE are not distinguishable from luck.
- [ ] 2x-cost, Wednesday-rebalance, random-half-universe and bi-weekly robustness checks completed.
- [ ] Each robustness variant reported with its matching equal-weight benchmark.
- [ ] Memorization probes on clean contexts near chance.
- [ ] Contamination gap below the pre-registered threshold.
- [ ] Leakage scan shows zero unexplained clean-period flags.
- [ ] `analysis_clean_test.{json,md}` reports every criterion, all metric CIs and
      pairwise comparisons; `claims_allowed` is true only if all pass.
- [ ] Top contributors checked against novel post-cutoff sources.
- [ ] Feasibility/100x-scale limitations stated.
- [ ] Alpha decay after the freeze reported; no live-trading claim.
- [ ] Final writeup says clearly whether the result is evidence of value or a
      well-measured failure to detect value.
