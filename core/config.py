"""Configuration loading shared by the whole project."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return ROOT


def load_yaml(path: str | Path) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(name: str) -> Any:
    """Load one of the YAML files in config/ by bare name."""
    return load_yaml(ROOT / "config" / name)


def load_universe(path: str | Path | None = None) -> dict[str, Any]:
    data = load_yaml(path or ROOT / "config" / "universe.yaml")
    groups: dict[str, str] = {}
    for group, tickers in (data.get("groups") or {}).items():
        for ticker in tickers:
            groups[str(ticker)] = str(group)
    data["group_map"] = groups
    tickers = sorted(groups)
    if len(tickers) != len(set(tickers)):
        raise ValueError("A ticker appears in more than one group in universe.yaml")
    data["tickers"] = tickers
    return data


def read_prompt(name: str) -> str:
    path = ROOT / "config" / "prompts" / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip() + "\n"


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(stable_json(obj))


def config_hash() -> str:
    """Hash of the frozen yardstick inputs (not source code)."""
    rel_paths = [
        "config/risk_limits.yaml",
        "config/costs.yaml",
        "config/tunables.yaml",
        "config/experiment.yaml",
        "config/periods.yaml",
        "config/models.yaml",
        "config/universe.yaml",
    ]
    pieces: list[str] = []
    for rel in rel_paths:
        p = ROOT / rel
        if p.exists():
            pieces.append(rel + ":" + hashlib.sha256(p.read_bytes()).hexdigest())
    prompts = sorted((ROOT / "config" / "prompts").glob("*.txt"))
    for p in prompts:
        pieces.append(
            str(p.relative_to(ROOT)) + ":" + hashlib.sha256(p.read_bytes()).hexdigest()
        )
    return sha256_text("\n".join(pieces))


def git_commit() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None
