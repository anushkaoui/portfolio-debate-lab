"""Freeze lock creation and verification.

The lock hashes prompts, configs, the universe, and the source of core/,
agents/, baselines/ and eval/. It is intentionally strict: any byte change
invalidates the lock. A lock is only valid for final evaluation when `approved`
and `model_cutoff_verified` are true.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import ROOT, git_commit, load_yaml  # noqa: E402

FREEZE_REL_PATHS = [
    "config/prompts",
    "config/risk_limits.yaml",
    "config/costs.yaml",
    "config/tunables.yaml",
    "config/tuning_grid.yaml",
    "config/experiment.yaml",
    "config/periods.yaml",
    "config/models.yaml",
    "config/universe.yaml",
    "core",
    "agents",
    "baselines",
    "eval",
]

EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store"}


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in FREEZE_REL_PATHS:
        p = root / rel
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
            continue
        for child in sorted(p.rglob("*")):
            if not child.is_file():
                continue
            if child.suffix in EXCLUDE_SUFFIXES:
                continue
            if any(part in EXCLUDE_PARTS for part in child.parts):
                continue
            files.append(child)
    return sorted(set(files))


def compute_freeze(root: str | Path | None = None) -> dict[str, Any]:
    root = Path(root or ROOT)
    files: dict[str, str] = {}
    for path in _iter_files(root):
        rel = str(path.relative_to(root))
        files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = hashlib.sha256(
        "\n".join(f"{rel}:{digest}" for rel, digest in sorted(files.items())).encode()
    ).hexdigest()

    models = load_yaml(root / "config" / "models.yaml").get("agent_model", {})
    experiment = load_yaml(root / "config" / "experiment.yaml")
    return {
        "freeze_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "combined_sha256": combined,
        "files": files,
        "model": {
            "provider": models.get("provider"),
            "name": models.get("name"),
            "pinned_snapshot": models.get("pinned_snapshot"),
            "documented_cutoff": models.get("documented_cutoff"),
            "cutoff_source_url": models.get("cutoff_source_url"),
            "checked_on": models.get("checked_on"),
            "verified": bool(models.get("verified")),
        },
        "model_cutoff_verified": bool(models.get("verified")),
        "experiment_id": experiment.get("experiment_id"),
        "git_commit": git_commit(),
        "approved": False,
        "approval_note": (
            "DRAFT. The human must review dev results, set approved: true (or run "
            "core/freeze.py --approve after verifying the model cutoff), and then "
            "run the final evaluation."
        ),
    }


def write_freeze_lock(
    path: str | Path | None = None,
    *,
    approve: bool = False,
    root: str | Path | None = None,
    approval_note: str | None = None,
) -> Path:
    root = Path(root or ROOT)
    path = Path(path or root / "freeze.lock")
    lock = compute_freeze(root)
    if approve:
        if not lock["model_cutoff_verified"]:
            raise ValueError(
                "Cannot approve freeze: models.yaml cutoff is not human-verified."
            )
        models = load_yaml(root / "config" / "models.yaml").get("agent_model", {})
        if not models.get("pinned_snapshot"):
            raise ValueError(
                "Cannot approve freeze: models.yaml must pin the exact served checkpoint."
            )
        if not models.get("cutoff_source_url"):
            raise ValueError(
                "Cannot approve freeze: models.yaml must record the cutoff source URL."
            )
        if not models.get("checked_on"):
            raise ValueError(
                "Cannot approve freeze: models.yaml must record when the cutoff was checked."
            )
        lock["approved"] = True
        lock["approval_note"] = approval_note or "Approved by human after dev review."
    else:
        lock["approved"] = False
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Create a git tag only if the caller asks and the tree has a commit.
    if approve and lock.get("git_commit"):
        import subprocess

        tag = f"freeze-{lock['combined_sha256'][:12]}"
        subprocess.run(
            ["git", "tag", "-f", tag],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    return path


def load_freeze_lock(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or ROOT / "freeze.lock")
    if not p.exists():
        raise FileNotFoundError(f"No freeze lock at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_freeze_lock(
    path: str | Path | None = None,
    *,
    root: str | Path | None = None,
    require_approved: bool = True,
) -> tuple[bool, str]:
    root = Path(root or ROOT)
    try:
        lock = load_freeze_lock(path or root / "freeze.lock")
    except FileNotFoundError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"unreadable lock: {exc}"

    current = compute_freeze(root)
    if lock.get("combined_sha256") != current["combined_sha256"]:
        changed = []
        old_files = lock.get("files") or {}
        new_files = current["files"]
        for rel in sorted(set(old_files) | set(new_files)):
            if old_files.get(rel) != new_files.get(rel):
                changed.append(rel)
        return False, "frozen files changed: " + ", ".join(changed[:10])
    if require_approved:
        if not lock.get("approved"):
            return False, "freeze lock is a DRAFT, not human-approved"
        if not lock.get("model_cutoff_verified"):
            return False, "freeze lock does not contain a verified model cutoff"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Create or verify freeze.lock")
    parser.add_argument("--approve", action="store_true", help="mark approved (requires verified cutoff)")
    parser.add_argument("--note", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        ok, reason = verify_freeze_lock(args.path)
        print(("OK: " if ok else "FAIL: ") + reason)
        return 0 if ok else 1
    path = write_freeze_lock(args.path, approve=args.approve, approval_note=args.note)
    print(f"wrote {path}")
    ok, reason = verify_freeze_lock(path, require_approved=False)
    print(f"integrity: {'OK' if ok else 'FAIL'} ({reason})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
