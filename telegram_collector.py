"""텔레그램 방 메시지 수집기 (내 계정으로 읽기, Telethon).

정해진 방의 새 메시지를 주기적으로(기본 5분) 가져와 SQLite에 쌓는다.
최근 메시지는 매번 다시 읽어 수정된 글과 조회수도 갱신한다.

준비 (한 번만):
  1. https://my.telegram.org → API development tools 에서 api_id / api_hash 발급
  2. .env 에 TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_CHAT_ID(수집할 방 id) 입력
     (.env 는 커밋되지 않음)
  3. 터미널에서 로그인:   .venv\\Scripts\\python.exe telegram_collector.py login
     (전화번호 → 텔레그램 앱으로 온 코드 → 2단계 비밀번호가 있으면 입력)

실행:
  .venv\\Scripts\\python.exe telegram_collector.py run          # 5분마다 계속
  .venv\\Scripts\\python.exe telegram_collector.py once         # 한 번만

로그인 세션 파일(data/telegram/session.session)은 계정 로그인과 같은
효력이 있으므로 남에게 주거나 저장소에 넣지 말 것.
"""
import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.default_config import TRADINGAGENTS_HOME
from typing import Any, Dict, Optional

from dotenv import load_dotenv

DATA_DIR = Path(TRADINGAGENTS_HOME) / "telegram"
DB_PATH = DATA_DIR / "messages.db"
SESSION_PATH = DATA_DIR / "session"  # Telethon이 .session 을 붙인다
DEFAULT_INTERVAL = 300
BACKFILL = 200  # 처음 실행할 때 가져올 과거 메시지 수
REFRESH = 50  # 매번 다시 읽어 수정·조회수를 갱신할 최근 메시지 수

log = logging.getLogger("telegram_collector")

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    chat_id      INTEGER NOT NULL,
    msg_id       INTEGER NOT NULL,
    date         TEXT    NOT NULL,
    edit_date    TEXT,
    sender_id    INTEGER,
    sender_name  TEXT,
    text         TEXT,
    views        INTEGER,
    forwards     INTEGER,
    replies      INTEGER,
    reply_to     INTEGER,
    media        TEXT,
    link         TEXT,
    collected_at TEXT    NOT NULL,
    PRIMARY KEY (chat_id, msg_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_date ON messages (chat_id, date);
CREATE TABLE IF NOT EXISTS chats (
    chat_id    INTEGER PRIMARY KEY,
    title      TEXT,
    username   TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    started_at  TEXT,
    finished_at TEXT,
    chat_id     INTEGER,
    new         INTEGER,
    updated     INTEGER,
    error       TEXT
);
"""

_COLUMNS = (
    "chat_id", "msg_id", "date", "edit_date", "sender_id", "sender_name", "text",
    "views", "forwards", "replies", "reply_to", "media", "link", "collected_at",
)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _now() -> str:
    return _iso(datetime.now(timezone.utc))


def message_link(chat_id: int, msg_id: int, username: Optional[str] = None) -> str:
    """공개 방은 t.me/<이름>/<번호>, 비공개 방은 t.me/c/<내부 id>/<번호> (방 멤버만 열림)."""
    if username:
        return f"https://t.me/{username}/{msg_id}"
    internal = str(abs(chat_id))
    if internal.startswith("100") and len(internal) > 10:
        internal = internal[3:]
    return f"https://t.me/c/{internal}/{msg_id}"


def _sender_name(sender: Any) -> Optional[str]:
    if sender is None:
        return None
    title = getattr(sender, "title", None)  # 채널 명의로 올린 글
    if title:
        return title
    name = " ".join(p for p in (getattr(sender, "first_name", None), getattr(sender, "last_name", None)) if p)
    return name or getattr(sender, "username", None)


def _media_kind(msg: Any) -> Optional[str]:
    media = getattr(msg, "media", None)
    if media is None:
        return None
    return type(media).__name__.replace("MessageMedia", "").lower() or "media"


def message_row(msg: Any, chat_id: int, username: Optional[str] = None) -> Dict[str, Any]:
    """Telethon Message → DB 행."""
    replies = getattr(msg, "replies", None)
    reply_to = getattr(msg, "reply_to", None)
    return {
        "chat_id": chat_id,
        "msg_id": msg.id,
        "date": _iso(msg.date),
        "edit_date": _iso(getattr(msg, "edit_date", None)),
        "sender_id": getattr(msg, "sender_id", None),
        "sender_name": _sender_name(getattr(msg, "sender", None)) or getattr(msg, "post_author", None),
        "text": getattr(msg, "message", None) or "",
        "views": getattr(msg, "views", None),
        "forwards": getattr(msg, "forwards", None),
        "replies": getattr(replies, "replies", None) if replies else None,
        "reply_to": getattr(reply_to, "reply_to_msg_id", None) if reply_to else None,
        "media": _media_kind(msg),
        "link": message_link(chat_id, msg.id, username),
        "collected_at": _now(),
    }


def open_db(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # 앱이 읽는 중에도 쓸 수 있게
    conn.executescript(SCHEMA)
    return conn


def upsert(conn: sqlite3.Connection, rows) -> tuple:
    """행들을 저장하고 (새로 추가된 수, 갱신된 수)를 반환한다."""
    new = updated = 0
    for row in rows:
        exists = conn.execute(
            "SELECT 1 FROM messages WHERE chat_id=? AND msg_id=?", (row["chat_id"], row["msg_id"])
        ).fetchone()
        conn.execute(
            f"INSERT OR REPLACE INTO messages ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
            [row[c] for c in _COLUMNS],
        )
        if exists:
            updated += 1
        else:
            new += 1
    conn.commit()
    return new, updated


def last_msg_id(conn: sqlite3.Connection, chat_id: int) -> int:
    row = conn.execute("SELECT MAX(msg_id) FROM messages WHERE chat_id=?", (chat_id,)).fetchone()
    return row[0] or 0


def read_messages(query: str = "", limit: int = 500, path: Path = DB_PATH):
    """앱 표시용: 최신순 메시지와 방 이름·최근 수집 기록. DB가 없으면 None."""
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        sql, args = "SELECT m.*, c.title FROM messages m LEFT JOIN chats c USING (chat_id)", []
        if query.strip():
            sql += " WHERE m.text LIKE ? OR m.sender_name LIKE ?"
            args += [f"%{query.strip()}%"] * 2
        sql += " ORDER BY m.date DESC LIMIT ?"
        messages = [dict(r) for r in conn.execute(sql, args + [limit])]
        total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        runs = [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY finished_at DESC LIMIT 5")]
        return {"messages": messages, "total": total, "runs": runs}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Telethon
# ---------------------------------------------------------------------------
def _client():
    from telethon import TelegramClient

    load_dotenv(Path(__file__).with_name(".env"), override=True)
    api_id, api_hash = os.getenv("TELEGRAM_API_ID"), os.getenv("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        sys.exit(
            "TELEGRAM_API_ID / TELEGRAM_API_HASH 가 .env 에 없습니다. "
            "https://my.telegram.org 의 API development tools 에서 발급받아 넣어 주세요."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return TelegramClient(str(SESSION_PATH), int(api_id), api_hash)


async def _resolve(client, chat_id: int):
    try:
        return await client.get_entity(chat_id)
    except ValueError:
        # 세션에 아직 방 정보가 없으면 대화 목록을 한 번 읽어 채운다.
        await client.get_dialogs()
        return await client.get_entity(chat_id)


async def collect_once(client, conn: sqlite3.Connection, chat_id: int) -> tuple:
    entity = await _resolve(client, chat_id)
    username = getattr(entity, "username", None)
    conn.execute(
        "INSERT OR REPLACE INTO chats (chat_id, title, username, updated_at) VALUES (?, ?, ?, ?)",
        (chat_id, getattr(entity, "title", None), username, _now()),
    )

    since = last_msg_id(conn, chat_id)
    if since:
        fresh = [m async for m in client.iter_messages(entity, min_id=since, reverse=True)]
    else:
        fresh = [m async for m in client.iter_messages(entity, limit=BACKFILL)]
    recent = [m async for m in client.iter_messages(entity, limit=REFRESH)]

    seen, rows = set(), []
    for m in fresh + recent:
        if m.id in seen or getattr(m, "action", None):  # 입장·고정 같은 서비스 메시지는 건너뜀
            continue
        seen.add(m.id)
        rows.append(message_row(m, chat_id, username))
    return upsert(conn, rows)


async def _run(chat_id: int, interval: int, once: bool) -> None:
    from telethon.errors import FloodWaitError

    client = _client()
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        sys.exit("아직 로그인하지 않았습니다. 먼저 터미널에서 `telegram_collector.py login` 을 실행하세요.")
    conn = open_db()
    log.info("수집 시작: 방 %s, %s초 간격, DB %s", chat_id, interval, DB_PATH)
    try:
        while True:
            started, error, new, updated = _now(), None, 0, 0
            try:
                new, updated = await collect_once(client, conn, chat_id)
                log.info("새 메시지 %d건, 갱신 %d건", new, updated)
            except FloodWaitError as exc:
                error = f"FloodWait {exc.seconds}s"
                log.warning("텔레그램 요청 제한: %s초 대기", exc.seconds)
                await asyncio.sleep(exc.seconds)
            except Exception as exc:  # 한 번 실패해도 다음 주기에 다시 시도
                error = f"{type(exc).__name__}: {exc}"
                log.exception("수집 실패")
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)", (started, _now(), chat_id, new, updated, error)
            )
            conn.commit()
            if once:
                break
            await asyncio.sleep(interval)
    finally:
        conn.close()
        await client.disconnect()


def normalize_phone(raw: str) -> str:
    """'010-1234-5678' 처럼 국내 형식으로 쳐도 텔레그램이 받는 '+821012345678' 로 바꾼다."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if digits.startswith("820"):  # '+82 010-…' 처럼 국내 번호의 0을 남긴 경우
        digits = "82" + digits[3:]
    if raw.strip().startswith("+") or digits.startswith("82"):
        return "+" + digits
    if digits.startswith("0"):
        return "+82" + digits[1:]
    return "+" + digits


def _ask_phone() -> str:
    phone = normalize_phone(input("전화번호 (예: 010-1234-5678 또는 +821012345678): "))
    print(f"→ {phone[:6]}****{phone[-2:]} 로 로그인 코드를 요청합니다. 텔레그램 앱을 확인하세요.")
    return phone


async def _backfill(chat_id: int, batch: int = 500) -> None:
    """저장된 가장 오래된 글보다 앞선 과거 글을 끝까지 가져온다 (한 번만 돌리면 됨)."""
    client = _client()
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        sys.exit("아직 로그인하지 않았습니다. 먼저 `telegram_collector.py login` 을 실행하세요.")
    conn = open_db()
    try:
        entity = await _resolve(client, chat_id)
        username = getattr(entity, "username", None)
        oldest = conn.execute("SELECT MIN(msg_id) FROM messages WHERE chat_id=?", (chat_id,)).fetchone()[0] or 0
        rows, total = [], 0
        # wait_time: 텔레그램 요청 제한에 걸리지 않게 100건 묶음 사이를 조금씩 쉰다.
        async for m in client.iter_messages(entity, max_id=oldest, wait_time=1):
            if getattr(m, "action", None):
                continue
            rows.append(message_row(m, chat_id, username))
            if len(rows) >= batch:
                total += upsert(conn, rows)[0]
                log.info("과거 글 %d건 저장 (가장 오래된 글 %s)", total, rows[-1]["date"])
                rows = []
        total += upsert(conn, rows)[0]
        log.info("과거 글 가져오기 완료: %d건", total)
    finally:
        conn.close()
        await client.disconnect()


async def _login() -> None:
    client = _client()
    # 전화번호·코드·2단계 비밀번호를 터미널에서 직접 묻는다
    await client.start(
        phone=_ask_phone,
        code_callback=lambda: input("텔레그램 앱으로 받은 로그인 코드: ").strip(),
    )
    me = await client.get_me()
    print(f"로그인 완료: {_sender_name(me)} — 세션 저장 위치 {SESSION_PATH}.session")
    await client.disconnect()


def main(argv=None) -> None:
    load_dotenv(Path(__file__).with_name(".env"), override=True)
    parser = argparse.ArgumentParser(description="텔레그램 방 메시지 수집기")
    parser.add_argument("command", choices=["login", "run", "once", "backfill"])
    env_chat = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    parser.add_argument("--chat", type=int, default=int(env_chat) if env_chat else None,
                        help="수집할 방 id (기본: .env 의 TELEGRAM_CHAT_ID)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="수집 간격(초), 기본 300")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command != "login" and args.chat is None:
        parser.error("수집할 방이 없습니다. .env 에 TELEGRAM_CHAT_ID=-100… 를 넣거나 --chat 으로 지정하세요.")
    if args.command == "login":
        asyncio.run(_login())
    elif args.command == "backfill":
        asyncio.run(_backfill(args.chat))
    else:
        asyncio.run(_run(args.chat, max(60, args.interval), once=args.command == "once"))


if __name__ == "__main__":
    main()
