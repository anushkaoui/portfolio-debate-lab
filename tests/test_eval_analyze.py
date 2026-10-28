from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eval.analyze import main as analyze_main


def _result(run_id, strategy, equity, sharpe, period="dev"):
    return {
        "run_id": run_id,
        "strategy": strategy,
        "period": period,
        "mode": "mock",
        "model": "code",
        "metrics": {
            "sharpe": sharpe,
            "annualized_return": 0.05,
            "annualized_volatility": 0.1,
            "max_drawdown": -0.1,
            "annualized_turnover": 0.2,
            "n_periods": len(equity) - 1,
            "provisional_periods": True,
        },
        "equity": [{"date": f"2023-01-{i + 1:02d}", "equity": v} for i, v in enumerate(equity)],
    }


def test_analyze_ignores_identified_and_prefers_canonical(tmp_path):
    eq = [100, 101, 100.5, 102, 101, 103]
    deb = [100, 100.2, 100.1, 100.4, 100.3, 100.6]
    (tmp_path / "eq.json").write_text(json.dumps(_result("eq", "equal_weight_buy_hold", eq, 0.5)), encoding="utf-8")
    canonical = _result("deb", "debate", deb, 0.2)
    canonical["spec"] = {"frequency": "weekly", "rebalance_weekday": 4}
    (tmp_path / "deb.json").write_text(json.dumps(canonical), encoding="utf-8")
    robustness = _result("deb_robust", "debate", deb, 0.9)
    robustness["spec"] = {"frequency": "biweekly", "rebalance_weekday": 2}
    (tmp_path / "deb_robust.json").write_text(json.dumps(robustness), encoding="utf-8")
    identified = _result("deb_id", "debate", deb, 0.8)
    identified["identified"] = True
    identified["spec"] = {"frequency": "weekly", "rebalance_weekday": 4}
    (tmp_path / "deb_id.json").write_text(json.dumps(identified), encoding="utf-8")
    rc = analyze_main(["--results-dir", str(tmp_path), "--period", "dev", "--n-boot", "50"])
    assert rc == 0
    payload = json.loads((tmp_path / "analysis_dev.json").read_text(encoding="utf-8"))
    assert payload["n_results"] == 2
    debate_rows = [r for r in payload["rows"] if r["strategy"] == "debate"]
    assert len(debate_rows) == 1
    assert debate_rows[0]["run_id"] == "deb"


def test_analyze_writes_table(tmp_path):
    eq = [100, 101, 100.5, 102, 101, 103]
    deb = [100, 100.2, 100.1, 100.4, 100.3, 100.6]
    (tmp_path / "eq.json").write_text(json.dumps(_result("eq", "equal_weight_buy_hold", eq, 0.5)), encoding="utf-8")
    (tmp_path / "deb.json").write_text(json.dumps(_result("deb", "debate", deb, 0.2)), encoding="utf-8")
    rc = analyze_main(["--results-dir", str(tmp_path), "--period", "dev", "--n-boot", "100"])
    assert rc == 0
    assert (tmp_path / "analysis_dev.json").exists()
    text = (tmp_path / "analysis_dev.md").read_text(encoding="utf-8")
    assert "debate" in text
    payload = json.loads((tmp_path / "analysis_dev.json").read_text(encoding="utf-8"))
    assert payload["n_results"] == 2
