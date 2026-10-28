from __future__ import annotations

from datetime import date

import pytest

from agents.mock_llm import MockProvider
from core.anonymize import AnonymizationLeak, Anonymizer, build_anonymizer_for_universe
from core.config import load_config, load_universe
from core.context import ContextBuilder
from core.periods import decision_dates_for_period, load_period_table
from core.timegate import TimeGate, load_price_file


@pytest.fixture(scope="module")
def dev_setup():
    universe = load_universe()
    prices = load_price_file()
    gate = TimeGate(prices, universe=universe["tickers"], max_visible_date=date(2023, 12, 29))
    anonymizer = build_anonymizer_for_universe(universe)
    builder = ContextBuilder(gate, anonymizer, universe, load_config("risk_limits.yaml"))
    as_of = decision_dates_for_period(load_period_table(), "dev")[10]
    return universe, anonymizer, builder, as_of


def test_mapping_is_deterministic_and_roundtrips():
    universe = load_universe()
    a = build_anonymizer_for_universe(universe)
    b = build_anonymizer_for_universe(universe)
    assert a.ticker_to_code() == b.ticker_to_code()
    for ticker in universe["tickers"]:
        assert a.ticker(a.code(ticker)) == ticker
        assert a.group_code(ticker).startswith("G")
        assert a.group_name(ticker)


def test_find_identifiers_detects_tickers_and_dates():
    a = build_anonymizer_for_universe(load_universe())
    found = a.find_identifiers("AAPL on 2023-05-05")
    assert any(x.startswith("ticker:") for x in found)
    assert any(x.startswith("date:") for x in found)


def test_primary_context_is_clean_and_has_no_absolute_prices(dev_setup):
    universe, anonymizer, builder, as_of = dev_setup
    packet = builder.build(as_of)
    anonymizer.assert_clean(packet.context)
    text = str(packet.context)
    assert "AAPL" not in text
    assert "2023-" not in text
    assert "last_close" not in packet.context["features"]["S01"]
    assert "price_index_26w" in packet.context["features"]["S01"]
    assert packet.context["time"]["as_of"] == "t-0"


def test_context_hash_changes_with_portfolio(dev_setup):
    universe, anonymizer, builder, as_of = dev_setup
    a = builder.build(as_of)
    b = builder.build(as_of)
    c = builder.build(as_of, portfolio_weights={universe["tickers"][0]: 0.05})
    assert a.context_hash == b.context_hash
    assert a.context_hash != c.context_hash


def test_identified_context_is_explicitly_contaminated(dev_setup):
    universe, anonymizer, builder, as_of = dev_setup
    packet = builder.build(as_of, identified=True)
    first = packet.context["features"][universe["tickers"][0]]
    assert first.get("last_close") is not None
    assert first.get("company")
    assert as_of.isoformat() in packet.context["time"]["as_of"]


def test_mock_provider_remains_ignorant(dev_setup):
    provider = MockProvider(seed=1)
    data, usage = provider.generate(
        role="identification_guess",
        system="s",
        user="u",
        payload={},
        params={},
    )
    assert data["recognizes_asset"] is False
    assert data["guessed_asset"] == "unknown"
    assert usage["cost_usd"] == 0.0
