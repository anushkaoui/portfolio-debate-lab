"""Single cached, logged, schema-validated LLM call path."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol

from pydantic import BaseModel, ValidationError

from agents.mock_llm import MockProvider
from core.config import ROOT, load_yaml, stable_json


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


@dataclass
class LLMResult:
    value: BaseModel
    cached: bool
    usage: Usage
    key: str
    model: str
    provider: str


class Provider(Protocol):
    name: str
    model: str

    def generate(
        self,
        *,
        role: str,
        system: str,
        user: str,
        payload: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> tuple[dict, dict]: ...


class CallBudget:
    def __init__(self, max_calls: int = 1000) -> None:
        self.max_calls = int(max_calls)
        self.used = 0

    def check(self) -> None:
        if self.used >= self.max_calls:
            raise BudgetExceeded(
                f"LLM call budget exhausted ({self.used}/{self.max_calls})"
            )

    def consume(self, n: int = 1) -> None:
        self.used += n

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)


class CacheStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                  key TEXT PRIMARY KEY,
                  created_utc TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  role TEXT NOT NULL,
                  as_of TEXT NOT NULL,
                  response_json TEXT NOT NULL,
                  prompt_tokens INTEGER NOT NULL DEFAULT 0,
                  completion_tokens INTEGER NOT NULL DEFAULT 0,
                  cost_usd REAL NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts_utc TEXT NOT NULL,
                  key TEXT NOT NULL,
                  role TEXT NOT NULL,
                  as_of TEXT NOT NULL,
                  provider TEXT NOT NULL,
                  model TEXT NOT NULL,
                  cached INTEGER NOT NULL,
                  attempt INTEGER NOT NULL,
                  prompt_tokens INTEGER NOT NULL DEFAULT 0,
                  completion_tokens INTEGER NOT NULL DEFAULT 0,
                  cost_usd REAL NOT NULL DEFAULT 0,
                  ok INTEGER NOT NULL,
                  error TEXT
                )
                """
            )

    def get(self, key: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response_json, prompt_tokens, completion_tokens, cost_usd "
                "FROM cache WHERE key=?",
                (key,),
            ).fetchone()
        if not row:
            return None
        return {
            "response": json.loads(row[0]),
            "prompt_tokens": row[1],
            "completion_tokens": row[2],
            "cost_usd": row[3],
        }

    def put(
        self,
        key: str,
        *,
        provider: str,
        model: str,
        role: str,
        as_of: str,
        response: dict,
        usage: Usage,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache
                (key, created_utc, provider, model, role, as_of, response_json,
                 prompt_tokens, completion_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    provider,
                    model,
                    role,
                    as_of,
                    json.dumps(response, sort_keys=True),
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cost_usd,
                ),
            )

    def log_call(
        self,
        *,
        key: str,
        role: str,
        as_of: str,
        provider: str,
        model: str,
        cached: bool,
        attempt: int,
        usage: Usage,
        ok: bool,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO calls
                (ts_utc, key, role, as_of, provider, model, cached, attempt,
                 prompt_tokens, completion_tokens, cost_usd, ok, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    key,
                    role,
                    as_of,
                    provider,
                    model,
                    int(cached),
                    attempt,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.cost_usd,
                    int(ok),
                    error,
                ),
            )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def _parse_json(text: str) -> dict:
    text = _strip_fences(text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise LLMError("LLM response is not a JSON object")
    return data


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        pricing: Mapping[str, float],
        timeout: int = 90,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.pricing = {
            "input": float(pricing.get("input", 0.0)),
            "output": float(pricing.get("output", 0.0)),
        }
        self.timeout = timeout

    def _cost(self, pin: int, pout: int) -> float:
        return (pin / 1_000_000.0) * self.pricing["input"] + (
            pout / 1_000_000.0
        ) * self.pricing["output"]

    def generate(
        self,
        *,
        role: str,
        system: str,
        user: str,
        payload: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> tuple[dict, dict]:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(params.get("temperature", 0.2)),
            "max_tokens": int(params.get("max_output_tokens", 900)),
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        request = urllib.request.Request(
            self.api_base + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise LLMError(f"HTTP {exc.code} from LLM API: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Network error calling LLM API: {exc}") from exc
        try:
            outer = json.loads(raw)
            content = outer["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LLMError(f"Malformed LLM API response: {raw[:1000]}") from exc
        usage_raw = outer.get("usage") or {}
        pin = int(usage_raw.get("prompt_tokens", 0) or 0)
        pout = int(usage_raw.get("completion_tokens", 0) or 0)
        usage = {
            "prompt_tokens": pin,
            "completion_tokens": pout,
            "cost_usd": self._cost(pin, pout),
        }
        return _parse_json(content), usage


class DeepSeekProvider(OpenAICompatibleProvider):
    name = "deepseek"


class CachedLLM:
    def __init__(
        self,
        provider: Provider,
        cache: CacheStore,
        budget: CallBudget,
        *,
        price_input_per_million: float = 0.0,
        price_output_per_million: float = 0.0,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.budget = budget
        self.price_input = float(price_input_per_million)
        self.price_output = float(price_output_per_million)
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_provider_calls = 0
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.log: list[dict] = []

    def _key(
        self,
        *,
        system: str,
        user: str,
        schema_name: str,
        params: Mapping[str, Any],
        as_of: date,
        context_hash: str,
    ) -> str:
        material = stable_json(
            {
                "provider": self.provider.name,
                "model": self.provider.model,
                "system": system,
                "user": user,
                "schema": schema_name,
                "params": params,
                "as_of": as_of.isoformat(),
                "context_hash": context_hash,
            }
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def call(
        self,
        *,
        role: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        params: Mapping[str, Any],
        as_of: date,
        context_hash: str,
        payload: Mapping[str, Any] | None = None,
    ) -> LLMResult:
        key = self._key(
            system=system,
            user=user,
            schema_name=schema.__name__,
            params=params,
            as_of=as_of,
            context_hash=context_hash,
        )
        self.n_calls += 1
        cached = self.cache.get(key)
        if cached is not None:
            try:
                value = schema.model_validate(cached["response"])
                usage = Usage(
                    int(cached["prompt_tokens"]),
                    int(cached["completion_tokens"]),
                    float(cached["cost_usd"]),
                )
                self.n_cache_hits += 1
                self.cache.log_call(
                    key=key,
                    role=role,
                    as_of=as_of.isoformat(),
                    provider=self.provider.name,
                    model=self.provider.model,
                    cached=True,
                    attempt=0,
                    usage=usage,
                    ok=True,
                )
                self.log.append(
                    {"role": role, "as_of": as_of.isoformat(), "cached": True, "key": key}
                )
                return LLMResult(value, True, usage, key, self.provider.model, self.provider.name)
            except ValidationError:
                # Corrupt/obsolete cache entry: treat as a miss.
                pass

        last_error: str | None = None
        for attempt in (1, 2):
            self.budget.check()
            self.budget.consume(1)
            self.n_provider_calls += 1
            try:
                raw, usage_raw = self.provider.generate(
                    role=role,
                    system=system,
                    user=user,
                    payload=payload or {},
                    params=params,
                )
                response = schema.model_validate(raw).model_dump(mode="json")
                usage = Usage(
                    int(usage_raw.get("prompt_tokens", 0)),
                    int(usage_raw.get("completion_tokens", 0)),
                    float(usage_raw.get("cost_usd", 0.0)),
                )
                self.total_cost += usage.cost_usd
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                self.cache.put(
                    key,
                    provider=self.provider.name,
                    model=self.provider.model,
                    role=role,
                    as_of=as_of.isoformat(),
                    response=response,
                    usage=usage,
                )
                self.cache.log_call(
                    key=key,
                    role=role,
                    as_of=as_of.isoformat(),
                    provider=self.provider.name,
                    model=self.provider.model,
                    cached=False,
                    attempt=attempt,
                    usage=usage,
                    ok=True,
                )
                self.log.append(
                    {
                        "role": role,
                        "as_of": as_of.isoformat(),
                        "cached": False,
                        "key": key,
                        "attempt": attempt,
                    }
                )
                return LLMResult(
                    schema.model_validate(response), False, usage, key, self.provider.model, self.provider.name
                )
            except (ValidationError, LLMError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                self.cache.log_call(
                    key=key,
                    role=role,
                    as_of=as_of.isoformat(),
                    provider=self.provider.name,
                    model=self.provider.model,
                    cached=False,
                    attempt=attempt,
                    usage=Usage(),
                    ok=False,
                    error=last_error[:1000],
                )
                if attempt == 1:
                    user = (
                        user
                        + "\n\nYour previous output was invalid JSON or violated the schema. "
                        "Return only a corrected JSON object. Error: "
                        + last_error[:500]
                    )
                    time.sleep(0.05)
        raise LLMError(f"LLM call failed after retry for role={role}: {last_error}")


def make_client(
    mode: str,
    *,
    models_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    max_calls: int = 1000,
    mock_seed: int = 0,
) -> CachedLLM:
    mode = mode.lower()
    cache = CacheStore(cache_path or (ROOT / "data" / "cache" / "llm_cache.sqlite"))
    budget = CallBudget(max_calls)
    if mode == "mock":
        return CachedLLM(MockProvider(seed=mock_seed), cache, budget)
    models = load_yaml(models_path or (ROOT / "config" / "models.yaml"))
    cfg = models.get("agent_model", {})
    provider_name = str(cfg.get("provider", "deepseek")).lower()
    api_key = os.environ.get(str(cfg.get("env_var", "DEEPSEEK_API_KEY")), "")
    if not api_key:
        raise LLMError(
            f"Missing API key in environment variable {cfg.get('env_var')}. "
            "Use --mock for offline runs."
        )
    pricing = cfg.get("pricing_usd_per_million") or {}
    if provider_name in {"deepseek", "openai", "openai-compatible"}:
        provider = DeepSeekProvider(
            model=str(cfg.get("name", "deepseek-chat")),
            api_base=str(cfg.get("api_base", "https://api.deepseek.com")),
            api_key=api_key,
            pricing=pricing,
        )
    else:
        raise LLMError(f"Unsupported provider {provider_name!r}")
    return CachedLLM(
        provider,
        cache,
        budget,
        price_input_per_million=float(pricing.get("input", 0.0)),
        price_output_per_million=float(pricing.get("output", 0.0)),
    )
