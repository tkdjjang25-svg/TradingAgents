"""Korean news from the Naver Search API (news), cleaned for an LLM.

Endpoint: GET https://naverapihub.apigw.ntruss.com/search/v1/news (NAVER API HUB)
with the X-NCP-APIGW-API-KEY-ID / X-NCP-APIGW-API-KEY headers carrying the HUB
application's Client ID / Client Secret (NAVER_CLIENT_ID / NAVER_CLIENT_SECRET, read
through ``get_env``; never logged or returned). NAVER_API_MODE=legacy switches to the
old Developers Center endpoint (openapi.naver.com, X-Naver-Client-Id/-Secret).

Contract
--------
``search_news`` -> list of dicts (newest first), each with ``title``, ``summary``,
``publisher``, ``published`` ("YYYY-MM-DD HH:MM", KST), ``url`` and ``similar``
(how many syndicated near-duplicates were merged into this article).
  * missing credentials -> ``[]``
  * HTTP 401/403, 429, other HTTP errors, timeouts -> raises ``NaverNewsError``
    (its message never contains the credentials). If an error happens on a later
    page after earlier pages succeeded, the articles already collected are returned.
  * a malformed start/end date -> ``ValueError``

``get_news_naver`` -> compact Markdown; never raises: configuration problems and
API errors come back as a one-line string.

Stateless and thread-safe: one module-level RateLimiter, a fresh request per call.
"""

import html
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests

from .external_common import RateLimiter, get_env

logger = logging.getLogger(__name__)

# Naver moved the Search API to NAVER Cloud's "NAVER API HUB" in 2026; new apps can only be
# created there. Old Developers Center apps keep working for a while: NAVER_API_MODE=legacy.
HUB_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
LEGACY_URL = "https://openapi.naver.com/v1/search/news.json"
API_URL = HUB_URL


def _endpoint(client_id: str, client_secret: str) -> Tuple[str, dict]:
    """(url, auth headers) for the configured API flavour (default: NAVER API HUB)."""
    if get_env("NAVER_API_MODE").lower() == "legacy":
        return LEGACY_URL, {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    return API_URL, {"X-NCP-APIGW-API-KEY-ID": client_id, "X-NCP-APIGW-API-KEY": client_secret}
KST = timezone(timedelta(hours=9))

_TIMEOUT = 10
_DISPLAY = 100  # API maximum per page
_MAX_PAGES = 3  # at most 300 articles / 3 requests per search
_RETRY_BACKOFF = 1.0  # seconds before the single retry
_SUMMARY_CHARS = 160
_DUP_RATIO = 0.85
_DUP_JACCARD = 0.7

_limiter = RateLimiter(0.12)


class NaverNewsError(Exception):
    """Naver API failure. ``status`` is the HTTP status (None for network errors)."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


# --- publishers -------------------------------------------------------------

_PUBLISHERS = {
    "hankyung.com": "한국경제",
    "mk.co.kr": "매일경제",
    "yna.co.kr": "연합뉴스",
    "yonhapnewstv.co.kr": "연합뉴스TV",
    "edaily.co.kr": "이데일리",
    "mt.co.kr": "머니투데이",
    "sedaily.com": "서울경제",
    "biz.chosun.com": "조선비즈",
    "chosun.com": "조선일보",
    "joongang.co.kr": "중앙일보",
    "donga.com": "동아일보",
    "hani.co.kr": "한겨레",
    "khan.co.kr": "경향신문",
    "newsis.com": "뉴시스",
    "news1.kr": "뉴스1",
    "etnews.com": "전자신문",
    "zdnet.co.kr": "지디넷코리아",
    "fnnews.com": "파이낸셜뉴스",
    "asiae.co.kr": "아시아경제",
    "heraldcorp.com": "헤럴드경제",
    "thebell.co.kr": "더벨",
    "infostockdaily.co.kr": "인포스탁데일리",
    "hankookilbo.com": "한국일보",
    "kmib.co.kr": "국민일보",
    "seoul.co.kr": "서울신문",
    "segye.com": "세계일보",
    "munhwa.com": "문화일보",
    "dt.co.kr": "디지털타임스",
    "ajunews.com": "아주경제",
    "inews24.com": "아이뉴스24",
    "bizwatch.co.kr": "비즈니스워치",
    "businesspost.co.kr": "비즈니스포스트",
    "newspim.com": "뉴스핌",
    "einfomax.co.kr": "연합인포맥스",
    "ddaily.co.kr": "디지털데일리",
    "etoday.co.kr": "이투데이",
    "nocutnews.co.kr": "노컷뉴스",
    "kbs.co.kr": "KBS",
    "imbc.com": "MBC",
    "sbs.co.kr": "SBS",
    "ytn.co.kr": "YTN",
    "jtbc.co.kr": "JTBC",
    "mbn.co.kr": "MBN",
    "sbsbiz.co.kr": "SBS Biz",
    "wowtv.co.kr": "한국경제TV",
    "hankyung.co.kr": "한국경제",
}
# Longest first so biz.chosun.com wins over chosun.com.
_PUBLISHER_KEYS = sorted(_PUBLISHERS, key=len, reverse=True)
_HOST_PREFIXES = ("www.", "m.", "mobile.")


def _host(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    for prefix in _HOST_PREFIXES:
        if host.startswith(prefix):
            host = host[len(prefix):]
    return host


def _publisher(originallink: str, link: str) -> str:
    host = _host(originallink)
    if not host:
        link_host = _host(link)
        if link_host.endswith("news.naver.com"):
            return "네이버뉴스"
        host = link_host
    for key in _PUBLISHER_KEYS:
        if host == key or host.endswith("." + key):
            return _PUBLISHERS[key]
    return host or "알 수 없음"


# --- text cleaning ----------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
# After unescaping, only strip real markup: "&lt;단독&gt;" is literal title text.
_ESCAPED_TAG = re.compile(r"</?(?:b|strong|em|i|u|br|span|p|font|a)(?:\s[^>]*)?/?>", re.IGNORECASE)
_WS = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Strip HTML tags, unescape entities (twice: Naver double-escapes some), collapse spaces."""
    text = _TAG.sub("", text or "")
    text = html.unescape(html.unescape(text))
    text = _ESCAPED_TAG.sub("", text)  # markup that was escaped in the source
    return _WS.sub(" ", text).strip()


_NOISE = [
    re.compile(r"\(\s*[가-힣A-Za-z]{1,10}\s*=\s*[^)]{1,20}\)"),  # (서울=연합뉴스)
    re.compile(r"\[[^\]]{0,40}(?:기자|특파원|에디터)[^\]]{0,20}\]"),  # [이데일리 홍길동 기자]
    re.compile(r"[가-힣]{2,4}\s*(?:기자|특파원)\s*=\s*"),  # 홍길동 기자 =
    re.compile(r"^\s*[가-힣]{2,4}\s*(?:기자|특파원)\s+"),  # leading "홍길동 기자 "
    re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),  # e-mails
    re.compile(r"(?:ⓒ|©|\(c\)|Copyright)[^.。]{0,60}", re.IGNORECASE),
    re.compile(r"[<\[(]?\s*저작권자[^>\])]{0,60}[>\])]?"),
    re.compile(r"무단\s*전재\s*(?:및|&|-)?\s*재배포\s*금지"),
    re.compile(r"재배포\s*금지"),
    re.compile(r"▶[^▶]*$"),  # trailing promo links
]
_LEADING_PUNCT = re.compile(r"^[\s=\-–—·:|/]+")


def _clean_summary(text: str) -> str:
    text = _clean_text(text)
    for pattern in _NOISE:
        text = pattern.sub(" ", text)
    text = _LEADING_PUNCT.sub("", _WS.sub(" ", text)).strip()
    return _trim(text, _SUMMARY_CHARS)


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.·…-") + "…"


# --- de-duplication ---------------------------------------------------------

_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)|【[^】]*】|〈[^〉]*〉|<[^>]*>|「[^」]*」")
_NON_WORD = re.compile(r"[^\w]+")


def _title_key(title: str) -> Tuple[str, frozenset]:
    """(compact normalised title, token set) — brackets like [단독]/(종합) removed."""
    base = _BRACKETS.sub(" ", title).lower()
    tokens = [t for t in _NON_WORD.split(base) if t and t != "_"]
    return "".join(tokens), frozenset(tokens)


def _similar(a: Tuple[str, frozenset], b: Tuple[str, frozenset]) -> bool:
    ca, ta = a
    cb, tb = b
    if not ca or not cb:
        return False
    if ca == cb:
        return True
    shorter = min(len(ca), len(cb))
    if shorter >= 12 and (ca in cb or cb in ca):
        return True
    if ta and tb and len(ta & tb) / len(ta | tb) >= _DUP_JACCARD:
        return True
    matcher = SequenceMatcher(None, ca, cb, autojunk=False)
    return matcher.real_quick_ratio() >= _DUP_RATIO and matcher.quick_ratio() >= _DUP_RATIO \
        and matcher.ratio() >= _DUP_RATIO


_BODY_MIN = 40  # summaries shorter than this are too generic to compare
_BODY_PREFIX = 50
_BODY_RATIO = 0.9


def _body_key(summary: str) -> str:
    """Compact summary text (no spaces/punctuation) for spotting the same body under a new title."""
    return "".join(t for t in _NON_WORD.split(summary.lower()) if t and t != "_")


def _same_body(a: str, b: str) -> bool:
    if len(a) < _BODY_MIN or len(b) < _BODY_MIN:
        return False
    if a[:_BODY_PREFIX] == b[:_BODY_PREFIX]:
        return True
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return matcher.real_quick_ratio() >= _BODY_RATIO and matcher.quick_ratio() >= _BODY_RATIO \
        and matcher.ratio() >= _BODY_RATIO


def _dedupe(articles: List[dict]) -> List[dict]:
    """Merge syndicated copies; keep each story's earliest (original) report.

    ``articles`` carry a private ``_dt`` (aware datetime). Exact URL repeats are
    dropped first. Two articles are the same story when their titles are near
    duplicates or their summaries are (one outlet re-titling the same body).
    Within a cluster the earliest article wins; ties go to the longer summary.
    Output is newest first.
    """
    seen_urls = set()
    clusters: List[dict] = []  # {"keys": [...], "bodies": [...], "members": [...]}
    for art in articles:
        if art["url"] and art["url"] in seen_urls:
            continue
        seen_urls.add(art["url"])
        key = _title_key(art["title"])
        body = _body_key(art["summary"])
        for cluster in clusters:
            if any(_similar(key, k) for k in cluster["keys"]) or \
                    any(_same_body(body, b) for b in cluster["bodies"]):
                cluster["keys"].append(key)
                cluster["bodies"].append(body)
                cluster["members"].append(art)
                break
        else:
            clusters.append({"keys": [key], "bodies": [body], "members": [art]})

    out = []
    for cluster in clusters:
        members = cluster["members"]
        best = min(members, key=lambda a: (a["_dt"], -len(a["summary"])))
        out.append({**best, "similar": len(members) - 1})
    out.sort(key=lambda a: a["_dt"], reverse=True)
    return out


# --- HTTP -------------------------------------------------------------------


def _credentials() -> Tuple[str, str]:
    return get_env("NAVER_CLIENT_ID"), get_env("NAVER_CLIENT_SECRET")


def _scrub(message: str, secrets: Tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***")
    return message


def _request(params: dict, client_id: str, client_secret: str) -> dict:
    url, headers = _endpoint(client_id, client_secret)
    secrets = (client_id, client_secret)
    for attempt in (1, 2):
        _limiter.wait()
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 1:
                time.sleep(_RETRY_BACKOFF)
                continue
            raise NaverNewsError(f"Naver News request failed ({type(exc).__name__}).") from None
        except requests.RequestException as exc:
            raise NaverNewsError(f"Naver News request failed ({type(exc).__name__}).") from None

        status = resp.status_code
        if status == 200:
            try:
                return resp.json()
            except ValueError:
                raise NaverNewsError("Naver News returned a non-JSON response.", status) from None
        if status in (429,) or status >= 500:
            if attempt == 1:
                time.sleep(_RETRY_BACKOFF * (2 if status == 429 else 1))
                continue
        if status in (401, 403):
            raise NaverNewsError(
                f"Naver News rejected the credentials (HTTP {status}); check NAVER_CLIENT_ID / "
                "NAVER_CLIENT_SECRET in .env and that the Search API is enabled for the app.",
                status,
            )
        if status == 429:
            raise NaverNewsError(
                "Naver News rate limit or daily quota exceeded (HTTP 429); try again later.", status
            )
        code = ""
        try:
            body = resp.json()
            if isinstance(body, dict) and body.get("errorCode"):
                code = f", errorCode {_scrub(str(body['errorCode'])[:10], secrets)}"
        except ValueError:
            pass
        raise NaverNewsError(f"Naver News request failed (HTTP {status}{code}).", status)
    raise NaverNewsError("Naver News request failed.")  # pragma: no cover


# --- public API -------------------------------------------------------------


def _parse_day(value: Optional[str], name: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").replace(tzinfo=KST)
    except ValueError:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {value!r}") from None


def _parse_pub(value: str) -> Optional[datetime]:
    try:
        dt = parsedate_to_datetime(value or "")
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.astimezone(KST)


def _convert(item: dict) -> Optional[dict]:
    dt = _parse_pub(item.get("pubDate", ""))
    title = _clean_text(item.get("title", ""))
    if dt is None or not title:
        return None
    originallink = (item.get("originallink") or "").strip()
    link = (item.get("link") or "").strip()
    return {
        "title": title,
        "summary": _clean_summary(item.get("description", "")),
        "publisher": _publisher(originallink, link),
        "published": dt.strftime("%Y-%m-%d %H:%M"),
        "url": originallink or link,
        "_dt": dt,
    }


def search_news(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 15,
) -> List[dict]:
    """Cleaned, de-duplicated Naver news for ``query``, newest first.

    Articles published after ``end_date`` 23:59:59 KST (the analysis date: no
    look-ahead) or before ``start_date`` 00:00 KST are dropped. Pages through
    the date-sorted results (100 per request, at most 3 requests) only while
    they are still newer than ``start_date``. Returns [] without credentials;
    raises NaverNewsError on API failures (see module docstring).
    """
    query = (query or "").strip()
    limit = max(1, min(int(limit or 15), 100))
    start_dt = _parse_day(start_date, "start_date")
    end_day = _parse_day(end_date, "end_date")
    end_dt = end_day + timedelta(days=1) if end_day else None  # exclusive bound
    client_id, client_secret = _credentials()
    if not query or not client_id or not client_secret:
        return []

    kept: List[dict] = []
    for page in range(_MAX_PAGES):
        params = {"query": query, "display": _DISPLAY, "start": 1 + page * _DISPLAY, "sort": "date"}
        try:
            data = _request(params, client_id, client_secret)
        except NaverNewsError:
            if kept:
                logger.warning("Naver News: page %d failed; returning earlier pages", page + 1)
                break
            raise
        items = data.get("items") or []
        oldest = None
        for item in items:
            art = _convert(item)
            if art is None:
                continue
            oldest = art["_dt"] if oldest is None else min(oldest, art["_dt"])
            if end_dt and art["_dt"] >= end_dt:
                continue
            if start_dt and art["_dt"] < start_dt:
                continue
            kept.append(art)
        total = int(data.get("total") or 0)
        if len(items) < _DISPLAY or page * _DISPLAY + len(items) >= min(total or 10**9, 1000):
            break  # no more results
        if start_dt and oldest is not None and oldest < start_dt:
            break  # already reached articles older than the window
        if sum(1 for a in _dedupe(kept) if _compact(query) in _compact(a["title"])) >= limit:
            break  # enough distinct stories about the query itself

    # One relevance-sorted page: Naver's own ranking of how central the query is.
    sim_rank = _relevance_page(query, client_id, client_secret, start_dt, end_dt, kept) \
        if _RELEVANCE_PAGE else {}

    articles = _select(_dedupe(kept), query, sim_rank, limit)
    for art in articles:
        art.pop("_dt", None)
    return articles


_SIM_TOP = 30
_RELEVANCE_PAGE = True  # one extra sort=sim request per search


def _relevance_page(query, client_id, client_secret, start_dt, end_dt, kept: List[dict]) -> dict:
    """Fetch Naver's relevance-sorted first page; add in-window articles to ``kept``.

    Returns {url: rank}. A failure only costs the relevance signal.
    """
    ranks = {}
    try:
        data = _request({"query": query, "display": _DISPLAY, "start": 1, "sort": "sim"},
                        client_id, client_secret)
    except NaverNewsError:
        logger.warning("Naver News: relevance page failed; using date order only")
        return ranks
    for rank, item in enumerate(data.get("items") or []):
        art = _convert(item)
        if art is None or (end_dt and art["_dt"] >= end_dt) or (start_dt and art["_dt"] < start_dt):
            continue
        ranks.setdefault(art["url"], rank)
        kept.append(art)
    return ranks


def _compact(text: str) -> str:
    return "".join(t for t in _NON_WORD.split((text or "").lower()) if t and t != "_")


def _relevance(art: dict, query: str, sim_rank: dict) -> int:
    """2 = query in the title, +1 = in the summary, +1 = in Naver's top relevance results."""
    q = _compact(query)
    score = 2 if q and q in _compact(art["title"]) else 0
    score += 1 if q and q in _compact(art["summary"]) else 0
    score += 1 if sim_rank.get(art["url"], _SIM_TOP) < _SIM_TOP else 0
    return score


def _select(articles: List[dict], query: str, sim_rank: dict, limit: int) -> List[dict]:
    """The ``limit`` most relevant stories (recency breaks ties), returned newest first.

    Passing mentions (query only somewhere in the body) are dropped unless there
    would otherwise be fewer than 3 articles.
    """
    scored = sorted(articles, key=lambda a: (_relevance(a, query, sim_rank), a["_dt"]), reverse=True)
    relevant = [a for a in scored if _relevance(a, query, sim_rank) > 0]
    chosen = (relevant if len(relevant) >= min(3, limit) else scored)[:limit]
    chosen.sort(key=lambda a: a["_dt"], reverse=True)
    return chosen


def get_news_naver(query: str, start_date: str, end_date: str, limit: int = 15) -> str:
    """``search_news`` as compact Markdown for an LLM; errors come back as a short string."""
    query = (query or "").strip()
    client_id, client_secret = _credentials()
    if not client_id or not client_secret:
        return ("Naver News is not configured (set NAVER_CLIENT_ID and NAVER_CLIENT_SECRET "
                "in .env).")
    if not query:
        return "Naver News: empty query."
    try:
        articles = search_news(query, start_date, end_date, limit)
    except ValueError as exc:
        return f"Naver News: {exc}"
    except NaverNewsError as exc:
        return _scrub(str(exc), (client_id, client_secret))
    except Exception as exc:  # never break the analyst on a data-source hiccup
        logger.exception("Naver News failed")
        return f"Naver News request failed ({type(exc).__name__})."

    window = f"{start_date} ~ {end_date}"
    if not articles:
        return f'No Naver News articles found for "{query}" ({window}).'
    lines = [f'## Naver News: "{query}" ({window})', ""]
    for art in articles:
        extra = f", +{art['similar']} similar" if art.get("similar") else ""
        lines.append(f"### {art['title']} ({art['publisher']}, {art['published']}{extra})")
        if art["summary"]:
            lines.append(art["summary"])
        if art["url"]:
            lines.append(art["url"])
        lines.append("")
    return _scrub("\n".join(lines).rstrip() + "\n", (client_id, client_secret))
