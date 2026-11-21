import sys
import unittest
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import history_store  # noqa: E402
from tradingagents.agents.utils.agent_utils import (  # noqa: E402
    build_instrument_context,
    build_position_context,
    format_price,
    normalize_position,
)
from tradingagents.graph.propagation import Propagator  # noqa: E402


@pytest.mark.unit
class TestPositionContext(unittest.TestCase):
    def test_missing_position_means_not_held(self):
        for pos in (None, {}, {"held": False, "avg_price": 100}):
            self.assertEqual(normalize_position(pos), {"held": False, "avg_price": None})
        text = build_position_context(None)
        self.assertIn("does NOT currently hold", text)
        self.assertIn("not trades", text)  # past decisions are not assumed executed

    def test_held_with_and_without_cost(self):
        text = build_position_context({"held": True, "avg_price": 210000})
        self.assertIn("currently HOLDS", text)
        self.assertIn("210,000", text)
        self.assertIn("not provided", build_position_context({"held": True, "avg_price": 0}))
        self.assertEqual(normalize_position({"held": True, "avg_price": "abc"})["avg_price"], None)

    def test_price_format(self):
        self.assertEqual(format_price(210000.0), "210,000")
        self.assertEqual(format_price(183.5), "183.5")

    def test_initial_state_defaults_to_not_held(self):
        state = Propagator().create_initial_state("NVDA", "2026-11-20")
        self.assertIn("does NOT currently hold", state["user_position"])
        ctx = build_instrument_context("NVDA", "", state["user_position"])
        self.assertIn("does NOT currently hold", ctx)
        held = Propagator().create_initial_state("NVDA", "2026-11-20", position={"held": True})
        self.assertIn("currently HOLDS", held["user_position"])


@pytest.mark.unit
class TestPositionDisplay(unittest.TestCase):
    def test_actions_depend_on_position(self):
        self.assertEqual(history_store.action_ko({"decision": "Underweight"}), "매수하지 않음")
        self.assertEqual(
            history_store.action_ko({"decision": "Underweight", "position": {"held": True}}),
            "일부 매도 (비중 축소)",
        )
        self.assertEqual(history_store.action_ko({"decision": "hold", "position": {"held": False}}), "매수 보류 · 관망")
        self.assertEqual(history_store.action_ko({"decision": "???"}), "")

    def test_position_labels(self):
        self.assertEqual(history_store.position_ko({}), "미입력 (미보유로 간주)")
        self.assertEqual(history_store.position_ko({"position": {"held": False}}), "미보유")
        self.assertEqual(
            history_store.position_ko({"position": {"held": True, "avg_price": 210000}}), "보유 중 · 평단 210,000"
        )


if __name__ == "__main__":
    unittest.main()
