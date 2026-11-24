import unittest
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.utils.agent_utils import build_instrument_context, build_user_notes
from tradingagents.dataflows import yfinance_news
from tradingagents.graph.propagation import Propagator


@pytest.mark.unit
class TestUserNotes(unittest.TestCase):
    def test_empty_notes_leave_prompts_unchanged(self):
        self.assertEqual(build_user_notes("  "), "")
        self.assertEqual(build_instrument_context("NVDA"), build_instrument_context("NVDA", ""))

    def test_notes_are_fenced_and_marked_unverified(self):
        section = build_user_notes(" 2027-01-12 평가기간 종료 ")
        self.assertIn("User-provided notes (unverified)", section)
        self.assertIn("<<<\n2027-01-12 평가기간 종료\n>>>", section)
        self.assertIn("2027-01-12", build_instrument_context("005930.KS", "2027-01-12"))

    def test_initial_state_carries_notes(self):
        state = Propagator().create_initial_state("005930.KS", "2026-11-21", user_notes=" memo ")
        self.assertEqual(state["user_notes"], "memo")
        self.assertEqual(Propagator().create_initial_state("NVDA", "2026-11-21")["user_notes"], "")

    def test_debaters_see_the_notes(self):
        llm = MagicMock()
        llm.invoke.return_value.content = "bull case"
        state = {
            "investment_debate_state": {"history": "", "bull_history": "", "current_response": "", "count": 0},
            "market_report": "m", "sentiment_report": "s", "news_report": "n", "fundamentals_report": "f",
            "user_notes": "buyback plan in late January",
        }
        create_bull_researcher(llm)(state)
        self.assertIn("buyback plan in late January", llm.invoke.call_args.args[0])


@pytest.mark.unit
class TestNewsFallback(unittest.TestCase):
    def test_company_query_drops_legal_suffixes(self):
        self.assertEqual(yfinance_news._company_query("Samsung Electronics Co., Ltd."), "Samsung Electronics")
        self.assertEqual(yfinance_news._company_query("NVIDIA Corporation"), "NVIDIA")
        self.assertEqual(yfinance_news._company_query("SK hynix Inc."), "SK hynix")

    def test_falls_back_to_search_by_company_name_when_get_news_is_down(self):
        article = {"title": "Samsung Stocks Fall", "publisher": "X", "link": "https://x",
                   "providerPublishTime": 1795210739}  # 2026-11-20 UTC
        by_symbol = MagicMock(news=[], quotes=[{"symbol": "005930.KS", "longname": "Samsung Electronics Co., Ltd."}])
        by_name = MagicMock(news=[article], quotes=[])
        ticker = MagicMock()
        ticker.get_news.side_effect = RuntimeError("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***")
        with patch.object(yfinance_news.yf, "Ticker", return_value=ticker), \
                patch.object(yfinance_news.yf, "Search", side_effect=[by_symbol, by_name]) as search:
            out = yfinance_news.get_news_yfinance("005930.KS", "2026-11-14", "2026-11-21")
        self.assertIn("### Samsung Stocks Fall (source: X)", out)
        self.assertEqual(search.call_args_list[1].args[0], "Samsung Electronics")

    def test_search_results_are_filtered_by_date(self):
        old = {"title": "Old", "providerPublishTime": 1700000000}  # 2023
        ticker = MagicMock()
        ticker.get_news.return_value = []
        with patch.object(yfinance_news.yf, "Ticker", return_value=ticker), \
                patch.object(yfinance_news.yf, "Search", return_value=MagicMock(news=[old], quotes=[])):
            out = yfinance_news.get_news_yfinance("NVDA", "2026-11-14", "2026-11-21")
        self.assertIn("No news found for NVDA between", out)


if __name__ == "__main__":
    unittest.main()
