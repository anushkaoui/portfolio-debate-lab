"""Deterministic mock model used by tests and offline dry runs.

It is NOT the research model. It implements plausible roles so the full
pipeline, risk layer, simulator and trace store can be exercised without API
calls. Every response is schema-valid and deterministic given the context.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


def _num(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _features(payload: Mapping[str, Any]) -> dict[str, dict]:
    ctx = payload.get("context") or {}
    return dict(ctx.get("features") or {})


def _current(payload: Mapping[str, Any]) -> dict[str, float]:
    ctx = payload.get("context") or {}
    return {str(k): _num(v) for k, v in (ctx.get("portfolio", {}).get("weights") or {}).items()}


def _codes(payload: Mapping[str, Any]) -> list[str]:
    ctx = payload.get("context") or {}
    return list(ctx.get("universe") or sorted(_features(payload)))


def _ev(code: str, key: str) -> str:
    return f"features.{code}.{key}"


def _trend_score(feat: Mapping[str, Any]) -> float:
    return (
        0.65 * _num(feat.get("ret_3m"))
        + 0.25 * _num(feat.get("ret_1m"))
        + 0.10 * _num(feat.get("ret_1w"))
    )


def _contrarian_score(feat: Mapping[str, Any]) -> float:
    return -(0.75 * _num(feat.get("ret_1m")) + 0.25 * _num(feat.get("ret_1w")))


def _make_proposals(payload: Mapping[str, Any], params: Mapping[str, Any], style: str) -> dict:
    feats = _features(payload)
    codes = [c for c in _codes(payload) if c in feats and not feats[c].get("missing")]
    if not codes:
        return {"proposals": [], "summary": "No usable assets in context.", "recognizes_asset": False}
    score_fn = _trend_score if style == "trend" else _contrarian_score
    ranked = sorted(codes, key=lambda c: score_fn(feats[c]), reverse=True)
    n = int(params.get("proposals_per_analyst", 3))
    base_delta = float(params.get("conviction_to_weight", 0.05))
    max_delta = float(params.get("max_analyst_delta", 0.08))
    proposals = []
    for code in ranked[:n]:
        feat = feats[code]
        score = score_fn(feat)
        if style == "trend":
            direction = "increase" if score > 0.01 else ("decrease" if score < -0.03 else "hold")
            evidence = [_ev(code, "ret_3m"), _ev(code, "ret_1m"), _ev(code, "rank_ret_3m")]
        else:
            direction = "increase" if score > 0.01 else ("decrease" if score < -0.03 else "hold")
            evidence = [_ev(code, "ret_1m"), _ev(code, "zscore_20d"), _ev(code, "rank_ret_1m")]
        conviction = max(0.35, min(0.9, 0.45 + abs(score)))
        delta = 0.0 if direction == "hold" else math.copysign(
            min(max_delta, base_delta * conviction), 1.0 if direction == "increase" else -1.0
        )
        proposals.append(
            {
                "agent": f"analyst_{style}",
                "asset": code,
                "direction": direction,
                "target_weight_delta": round(delta, 6),
                "conviction": round(conviction, 4),
                "horizon_days": 21 if style == "contrarian" else 63,
                "thesis": (
                    f"{style} signal score {score:.4f}; capped by {max_delta:.2f} per-name delta limit."
                ),
                "evidence": evidence,
                "what_would_change_my_mind": (
                    "A reversal in the cited return/rank evidence or a high-severity risk objection."
                ),
                "recognizes_asset": False,
            }
        )
    return {
        "proposals": proposals,
        "summary": f"{style} analyst generated {len(proposals)} proposals from numeric context.",
        "recognizes_asset": False,
    }


def _critiques(payload: Mapping[str, Any]) -> dict:
    feats = _features(payload)
    current = _current(payload)
    proposals = payload.get("proposals") or []
    critiques = []
    for prop in proposals:
        code = str(prop.get("asset", ""))
        feat = feats.get(code, {})
        delta = _num(prop.get("target_weight_delta"))
        cap = _num(feat.get("position_cap"), 0.10)
        vol = _num(feat.get("vol_20d"))
        target = current.get(code, 0.0) + delta
        objections: list[str] = []
        severity = "low"
        suggested = None
        if target > cap + 1e-9:
            severity = "high"
            suggested = cap
            objections.append(
                f"target {target:.4f} exceeds the volatility-scaled cap {cap:.4f} ({_ev(code, 'position_cap')})"
            )
        if vol > 0.35:
            severity = "high" if severity == "high" else "medium"
            objections.append(f"20-day volatility {vol:.4f} is elevated ({_ev(code, 'vol_20d')})")
        if abs(delta) > 0.06:
            severity = "high" if severity == "high" else "medium"
            objections.append(f"requested delta {delta:.4f} is large relative to the cap")
        if _num(feat.get("missing")):
            severity = "high"
            suggested = 0.0
            objections.append("asset data is missing or stale")
        if not objections:
            objections.append("no material risk objection from the supplied numbers")
        critiques.append(
            {
                "target": code,
                "objections": objections[:4],
                "severity": severity,
                "suggested_cap": round(suggested, 6) if suggested is not None else None,
                "evidence": [_ev(code, "vol_20d"), _ev(code, "position_cap")],
                "recognizes_asset": False,
            }
        )
    return {
        "critiques": critiques,
        "summary": "Risk manager checked position caps, volatility and missing data.",
        "recognizes_asset": False,
    }


def _responses(payload: Mapping[str, Any]) -> dict:
    proposals = payload.get("proposals") or []
    critiques = payload.get("critiques") or []
    by_target = {}
    for c in critiques:
        by_target.setdefault(str(c.get("target")), []).append(c)
    current = _current(payload)
    responses = []
    for prop in proposals:
        code = str(prop.get("asset", ""))
        delta = _num(prop.get("target_weight_delta"))
        severe = [c for c in by_target.get(code, []) if c.get("severity") == "high"]
        accept = bool(severe)
        revised = delta
        if severe:
            caps = [c.get("suggested_cap") for c in severe if c.get("suggested_cap") is not None]
            if caps:
                revised = min(delta, float(min(caps)) - current.get(code, 0.0))
            else:
                revised = delta * 0.5
            revised = max(-1.0, min(1.0, revised))
        responses.append(
            {
                "target": code,
                "accept_critique": accept,
                "revised_delta": round(revised, 6),
                "argument": (
                    "Accepting the high-severity cap and reducing the delta."
                    if accept
                    else "Contesting the critique: the supplied numeric headroom still supports the proposal."
                ),
                "evidence": [_ev(code, "position_cap"), _ev(code, "vol_20d")],
                "recognizes_asset": False,
            }
        )
    return {"responses": responses, "recognizes_asset": False}


def _normalise_weights(current: Mapping[str, float], codes: list[str]) -> dict[str, float]:
    weights = {c: max(0.0, float(current.get(c, 0.0))) for c in codes}
    total = sum(weights.values())
    if total > 1.0:
        weights = {c: v / total for c, v in weights.items()}
    rounded = {c: round(v, 6) for c, v in weights.items()}
    excess = sum(rounded.values()) - 1.0
    if excess > 0:
        largest = max(rounded, key=rounded.get)
        rounded[largest] = max(0.0, round(rounded[largest] - excess, 6))
    return rounded


def _decision(payload: Mapping[str, Any]) -> dict:
    codes = _codes(payload)
    current = {c: _current(payload).get(c, 0.0) for c in codes}
    proposals = payload.get("proposals") or []
    responses = {str(r.get("target")): r for r in (payload.get("responses") or [])}
    critiques = payload.get("critiques") or []
    has_high = any(c.get("severity") == "high" for c in critiques)
    evidence = ["portfolio.weights", "risk_headroom.max_single_position"]
    for prop in proposals:
        code = str(prop.get("asset", ""))
        if code not in current:
            continue
        delta = _num(prop.get("target_weight_delta"))
        if code in responses:
            resp = responses[code]
            if resp.get("revised_delta") is not None:
                delta = _num(resp.get("revised_delta"))
            evidence.append(_ev(code, "position_cap"))
        current[code] = max(0.0, current.get(code, 0.0) + delta)
    total = sum(current.values())
    if total > 1.0 and total > 0:
        current = {c: v / total for c, v in current.items()}
    # The mock trader respects the single-name cap for its own proposal, while
    # the hard risk layer remains the final authority.
    feats = _features(payload)
    for code in codes:
        cap = _num(feats.get(code, {}).get("position_cap"), 0.12)
        current[code] = min(current[code], cap)
    current = _normalise_weights(current, codes)
    return {
        "target_weights": current,
        "rationale": (
            "Combined numeric trend/contrarian deltas with the risk manager's caps. "
            + ("High-severity objections were reduced. " if has_high else "No high-severity objection. ")
            + "Hard risk layer applied downstream."
        ),
        "dissent_noted": has_high,
        "evidence": evidence[:12],
        "recognizes_asset": False,
    }


def mock_generate(role: str, payload: Mapping[str, Any], params: Mapping[str, Any]) -> dict:
    if role == "analyst_trend":
        return _make_proposals(payload, params, "trend")
    if role == "analyst_contrarian":
        return _make_proposals(payload, params, "contrarian")
    if role == "risk_manager":
        return _critiques(payload)
    if role == "analyst_response":
        return _responses(payload)
    if role == "trader":
        return _decision(payload)
    if role == "identification_guess":
        return {
            "recognizes_asset": False,
            "guessed_asset": "unknown",
            "guessed_period": "unknown",
            "confidence": 0.0,
            "reasoning": "The context is anonymized; no identity or date is present.",
        }
    if role == "probe_prediction":
        features = _features(payload)
        codes = [c for c in features if not features[c].get("missing")]
        if not codes:
            move = "unknown"
        else:
            avg = sum(_trend_score(features[c]) for c in codes) / len(codes)
            move = "up" if avg > 0.02 else ("down" if avg < -0.02 else "flat")
        return {
            "next_move": move,
            "confidence": 0.2,
            "rationale": "Weak deterministic prior from the supplied numeric features.",
            "recognizes_asset": False,
        }
    raise ValueError(f"Mock does not implement role {role!r}")


class MockProvider:
    name = "mock"
    model = "deterministic-mock-v1"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def generate(
        self,
        *,
        role: str,
        system: str,
        user: str,
        payload: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> tuple[dict, dict]:
        data = mock_generate(role, payload, params)
        return data, {"prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
