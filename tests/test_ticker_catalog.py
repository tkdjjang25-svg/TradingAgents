import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ticker_catalog  # noqa: E402


class _FakeSearch:
    def __init__(self, query, **kwargs):
        self.quotes = [
            {"symbol": "005930.KS", "shortname": "SamsungElec", "quoteType": "EQUITY", "exchDisp": "Korea"},
            {"symbol": "BTC=F", "shortname": "Bitcoin Futures", "quoteType": "FUTURE"},
            {"symbol": "BTC-USD", "shortname": "Bitcoin USD", "quoteType": "CRYPTOCURRENCY", "exchDisp": "CCC"},
        ]


@pytest.mark.unit
class TestTickerCatalog(unittest.TestCase):
    def test_search_keeps_analyzable_types_only(self):
        with patch("yfinance.Search", _FakeSearch):
            results = ticker_catalog.search("x")
        self.assertEqual([s for s, _ in results], ["005930.KS", "BTC-USD"])
        self.assertEqual(results[0][1], "SamsungElec · 주식 · Korea")

    def test_search_failure_and_blank_query_return_empty(self):
        self.assertEqual(ticker_catalog.search("   "), [])
        with patch("yfinance.Search", side_effect=RuntimeError("offline")):
            self.assertEqual(ticker_catalog.search("nvda"), [])

    def test_merge_keeps_order_and_first_label(self):
        merged = ticker_catalog.merge([("A", "search")], [("B", "recent"), ("A", "recent")], [("C", "")])
        self.assertEqual(list(merged), ["A", "B", "C"])
        self.assertEqual(merged["A"], "search")

    def test_ticker_validation(self):
        for ok in ("nvda", " 005930.ks ", "BTC-USD", "^GSPC", "BRK-B", "KRW=X"):
            self.assertTrue(ticker_catalog.is_valid_ticker(ok), ok)
        for bad in ("", "삼성전자", "apple inc", "A" * 21):
            self.assertFalse(ticker_catalog.is_valid_ticker(bad), bad)

    def test_popular_list_has_unique_valid_tickers(self):
        symbols = [s for s, _ in ticker_catalog.POPULAR]
        self.assertEqual(len(symbols), len(set(symbols)))
        self.assertTrue(all(ticker_catalog.is_valid_ticker(s) for s in symbols))
