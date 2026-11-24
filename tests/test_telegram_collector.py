import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import telegram_collector as tc  # noqa: E402


def _msg(msg_id, text="hello", **kw):
    base = dict(
        id=msg_id, date=datetime(2026, 11, 21, 1, 2, 3, tzinfo=timezone.utc), edit_date=None,
        sender_id=7, sender=SimpleNamespace(first_name="Min", last_name="Su", title=None, username="ms"),
        message=text, views=10, forwards=1, replies=None, reply_to=None, media=None, post_author=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.unit
class TestTelegramCollector(unittest.TestCase):
    def test_links_for_private_and_public_chats(self):
        self.assertEqual(tc.message_link(-1001234567890, 42), "https://t.me/c/1234567890/42")
        self.assertEqual(tc.message_link(-1001234567890, 42, "stocknews"), "https://t.me/stocknews/42")

    def test_korean_phone_numbers_get_the_country_code(self):
        for raw in ("010-1234-5678", "01012345678", "+82 10 1234 5678", "821012345678", "+82 010-1234-5678", "8201012345678"):
            self.assertEqual(tc.normalize_phone(raw), "+821012345678", raw)
        self.assertEqual(tc.normalize_phone("+1 415 555 0100"), "+14155550100")

    def test_message_row_fields(self):
        row = tc.message_row(_msg(5, reply_to=SimpleNamespace(reply_to_msg_id=3)), -1001234567890)
        self.assertEqual(row["date"], "2026-11-21T01:02:03+00:00")
        self.assertEqual((row["sender_name"], row["text"], row["reply_to"]), ("Min Su", "hello", 3))
        channel_post = tc.message_row(_msg(6, sender=SimpleNamespace(title="Test 채널")), -1001234567890)
        self.assertEqual(channel_post["sender_name"], "Test 채널")

    def test_upsert_counts_new_and_updated_and_read_back(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "m.db"
            conn = tc.open_db(path)
            chat = -1001234567890
            self.assertEqual(tc.upsert(conn, [tc.message_row(_msg(i), chat) for i in (1, 2)]), (2, 0))
            self.assertEqual(tc.upsert(conn, [tc.message_row(_msg(2, "edited"), chat),
                                              tc.message_row(_msg(3, "삼성전자 주주환원"), chat)]), (1, 1))
            self.assertEqual(tc.last_msg_id(conn, chat), 3)
            conn.close()

            data = tc.read_messages(path=path)
            self.assertEqual(data["total"], 3)
            self.assertEqual(tc.read_messages("주주환원", path=path)["messages"][0]["msg_id"], 3)
            self.assertEqual(tc.read_messages("edited", path=path)["messages"][0]["msg_id"], 2)
            self.assertIsNone(tc.read_messages(path=Path(tmp) / "missing.db"))


if __name__ == "__main__":
    unittest.main()
