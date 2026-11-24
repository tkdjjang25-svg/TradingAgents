"""SEC EDGAR client: US company filings (10-K, 10-Q, 8-K, ...) for the analysts.

Data sources (all free, no API key):
- https://www.sec.gov/files/company_tickers.json  ticker -> CIK map (cached on disk, 7 days)
- https://data.sec.gov/submissions/CIK##########.json  filing history (cached in memory, 10 min)
- https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<doc>  filing documents

SEC fair-access rules: every request carries the User-Agent from SEC_EDGAR_USER_AGENT
("AppName contact@email"; SEC answers 403 without one) and the whole process stays
below 10 requests/second. Nothing is requested when the setting is missing.

Every public function returns a short string (or None for get_cik) instead of raising.
"""

import html
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from .external_common import RateLimiter, cache_dir, get_env, market_of

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

MIN_INTERVAL = 0.12  # seconds between requests -> < 10 req/s, shared by all threads
TIMEOUT = 15
TICKERS_MAX_AGE = 7 * 24 * 3600
SUBMISSIONS_TTL = 600
BACKOFF_429 = 2.0
EXCERPT_CHARS = 600
MAX_OLDER_PAGES = 2

NOT_CONFIGURED = (
    'SEC EDGAR is not configured: set SEC_EDGAR_USER_AGENT="AppName your@email" in .env '
    "(SEC requires a descriptive User-Agent)."
)

_LIMITER = RateLimiter(MIN_INTERVAL)
_sleep = time.sleep  # indirection so tests can skip real back-off waits


class _SecError(Exception):
    """Internal: a short, user-presentable reason a request failed."""


# --------------------------------------------------------------------------- HTTP


DEFAULT_APP_NAME = "TradingAgentsKR"


def _user_agent() -> str:
    """User-Agent from the setting; SEC rejects a bare e-mail, so an app name is prefixed."""
    ua = get_env("SEC_EDGAR_USER_AGENT").strip().strip('"').strip("'")
    if ua and len(ua.split()) == 1 and "@" in ua:
        ua = f"{DEFAULT_APP_NAME} {ua}"
    return ua


def _http_get(url: str) -> requests.Response:
    """GET with the SEC headers, shared rate limit and one retry. Raises _SecError."""
    ua = _user_agent()
    if not ua:
        raise _SecError(NOT_CONFIGURED)
    headers = {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
    }
    last = ""
    for attempt in range(2):
        _LIMITER.wait()
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        except requests.Timeout:
            last = "SEC EDGAR request timed out"
            continue
        except requests.RequestException as exc:
            last = f"SEC EDGAR request failed ({type(exc).__name__})"
            continue
        status = resp.status_code
        if status == 200:
            return resp
        if status == 403:
            raise _SecError(
                "SEC EDGAR returned 403 Forbidden - usually a missing or non-descriptive "
                'User-Agent; set SEC_EDGAR_USER_AGENT="AppName your@email" in .env.'
            )
        if status == 404:
            raise _SecError("SEC EDGAR: not found (404).")
        if status == 429:
            last = "SEC EDGAR rate limit hit (429); try again shortly"
            if attempt == 0:
                _sleep(BACKOFF_429)
            continue
        if status >= 500:
            last = f"SEC EDGAR server error ({status})"
            continue
        raise _SecError(f"SEC EDGAR returned HTTP {status}.")
    raise _SecError(last + ".")


def _get_json(url: str):
    resp = _http_get(url)
    try:
        return resp.json()
    except ValueError:
        raise _SecError("SEC EDGAR returned invalid JSON.")


# --------------------------------------------------------------------------- ticker -> CIK

_tickers_lock = threading.Lock()
_tickers: Optional[Dict[str, dict]] = None
_tickers_time = 0.0  # wall-clock time the map was fetched (file mtime when loaded from disk)


def _normalise_ticker(ticker: str) -> str:
    return (ticker or "").strip().upper().replace(".", "-")


def _build_map(raw) -> Dict[str, dict]:
    rows = raw.values() if isinstance(raw, dict) else raw
    out: Dict[str, dict] = {}
    for row in rows:
        try:
            t = _normalise_ticker(row["ticker"])
            entry = {"cik": int(row["cik_str"]), "ticker": t, "title": str(row.get("title", ""))}
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault(t, entry)  # first entry wins (SEC lists the primary class first)
    return out


def _tickers_path():
    return cache_dir("sec_edgar") / "company_tickers.json"


def _write_atomic(path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".company_tickers.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _ticker_map() -> Dict[str, dict]:
    """The SEC ticker map: memory -> disk (< 7 days) -> network. Raises _SecError."""
    global _tickers, _tickers_time
    with _tickers_lock:
        now = time.time()
        if _tickers is not None and now - _tickers_time < TICKERS_MAX_AGE:
            return _tickers
        path = _tickers_path()
        stale = None
        if path.exists():
            try:
                mtime = path.stat().st_mtime
                loaded = _build_map(json.loads(path.read_text(encoding="utf-8")))
                if now - mtime < TICKERS_MAX_AGE and loaded:
                    _tickers, _tickers_time = loaded, mtime
                    return _tickers
                stale = (loaded, mtime)
            except (OSError, ValueError):
                stale = None
        try:
            raw = _get_json(TICKERS_URL)
            loaded = _build_map(raw)
            if not loaded:
                raise _SecError("SEC EDGAR ticker list was empty.")
        except _SecError:
            if stale and stale[0]:  # an old map beats no map
                _tickers, _tickers_time = stale[0], now - TICKERS_MAX_AGE + 3600  # retry in 1 h
                return _tickers
            raise
        _write_atomic(path, json.dumps(raw).encode("utf-8"))
        _tickers, _tickers_time = loaded, now
        return _tickers


def _lookup(ticker: str) -> Optional[dict]:
    return _ticker_map().get(_normalise_ticker(ticker))


def get_cik(ticker: str) -> Optional[dict]:
    """{"cik": int, "ticker": str, "title": str} for a US ticker, or None.

    Share classes use SEC's dash form (BRK.B -> BRK-B). None when the ticker is not
    in the SEC list (many ETFs are not), when SEC EDGAR is not configured and nothing
    is cached, or when the list cannot be fetched.
    """
    try:
        entry = _lookup(ticker)
    except _SecError:
        return None
    return dict(entry) if entry else None


# --------------------------------------------------------------------------- submissions

_sub_lock = threading.Lock()
_sub_cache: Dict[str, Tuple[float, dict]] = {}
_sub_key_locks: Dict[str, threading.Lock] = {}


def _cached_json(url: str) -> dict:
    """GET JSON with a 10-minute in-memory cache; concurrent callers share one fetch."""
    with _sub_lock:
        hit = _sub_cache.get(url)
        if hit and time.monotonic() - hit[0] < SUBMISSIONS_TTL:
            return hit[1]
        key_lock = _sub_key_locks.setdefault(url, threading.Lock())
    with key_lock:
        with _sub_lock:
            hit = _sub_cache.get(url)
            if hit and time.monotonic() - hit[0] < SUBMISSIONS_TTL:
                return hit[1]
        data = _get_json(url)
        if not isinstance(data, dict):
            raise _SecError("SEC EDGAR returned an unexpected response.")
        with _sub_lock:
            _sub_cache[url] = (time.monotonic(), data)
        return data


def _rows(block: dict) -> List[dict]:
    """Parallel arrays (form, filingDate, ...) -> list of dicts."""
    keys = ("form", "filingDate", "reportDate", "accessionNumber", "primaryDocument",
            "primaryDocDescription", "items")
    cols = {k: block.get(k) or [] for k in keys}
    n = len(cols["form"])
    out = []
    for i in range(n):
        out.append({k: (cols[k][i] if i < len(cols[k]) else "") or "" for k in keys})
    return out


def _filings_in_range(cik: int, start: str, end: str) -> Tuple[List[dict], str]:
    """Filings with start <= filingDate <= end, newest first, plus a coverage note."""
    sub = _cached_json(SUBMISSIONS_URL.format(cik=cik))
    filings = sub.get("filings") or {}
    rows = _rows(filings.get("recent") or {})
    note = ""
    oldest = min((r["filingDate"] for r in rows if r["filingDate"]), default="")
    if oldest and start < oldest:
        pages = [p for p in (filings.get("files") or [])
                 if isinstance(p, dict) and p.get("name")
                 and str(p.get("filingTo", "9999")) >= start and str(p.get("filingFrom", "")) <= end]
        for page in pages[:MAX_OLDER_PAGES]:
            try:
                rows += _rows(_cached_json(SUBMISSIONS_PAGE_URL.format(name=page["name"])))
            except _SecError:
                note = f"(Older filing history could not be fetched; coverage starts {oldest}.)"
                break
        if len(pages) > MAX_OLDER_PAGES:
            note = "(Very long history: only part of the older filings were scanned.)"
    picked = [r for r in rows if r["filingDate"] and start <= r["filingDate"] <= end]
    seen, unique = set(), []
    for r in picked:
        if r["accessionNumber"] not in seen:
            seen.add(r["accessionNumber"])
            unique.append(r)
    unique.sort(key=lambda r: r["filingDate"], reverse=True)
    return unique, note


# --------------------------------------------------------------------------- form vocabulary

ITEM_8K = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety",
    "1.05": "Material Cybersecurity Incident",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Event Accelerating a Financial Obligation",
    "2.05": "Exit or Disposal Activities (restructuring costs)",
    "2.06": "Material Impairments",
    "3.01": "Delisting Notice / Failure to Meet Listing Standards",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Change in Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.01": "Change in Control",
    "5.02": "Departure/Appointment of Directors or Officers; Compensation",
    "5.03": "Amendments to Articles/Bylaws; Fiscal Year Change",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Code of Ethics Amendment or Waiver",
    "5.06": "Change in Shell Company Status",
    "5.07": "Shareholder Vote Results",
    "5.08": "Shareholder Director Nominations",
    "6.01": "ABS Informational and Computational Material",
    "6.02": "Change of ABS Servicer or Trustee",
    "6.03": "Change in ABS Credit Enhancement",
    "6.04": "Failure to Make a Required ABS Distribution",
    "6.05": "ABS Securities Act Updating Disclosure",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

_FORM_DESC = {
    "10-K": "Annual report",
    "10-Q": "Quarterly report",
    "8-K": "Current report",
    "6-K": "Foreign issuer current report",
    "20-F": "Foreign issuer annual report",
    "40-F": "Canadian issuer annual report",
    "S-1": "Registration statement (IPO/offering)",
    "S-3": "Shelf registration",
    "S-3ASR": "Automatic shelf registration",
    "S-4": "Registration for merger/exchange offer",
    "F-1": "Foreign issuer registration statement",
    "F-3": "Foreign issuer shelf registration",
    "SC 13D": "Activist >5% ownership stake",
    "SC 13G": "Passive >5% ownership stake",
    "SCHEDULE 13D": "Activist >5% ownership stake",
    "SCHEDULE 13G": "Passive >5% ownership stake",
    "DEF 14A": "Proxy statement",
}

_INSIDER_FORMS = {"3", "4", "5"}
_PROSPECTUS_CAP = 3  # banks file hundreds of 424B2 note supplements; list a few, count the rest


def _base_form(form: str) -> str:
    return form[:-2] if form.endswith("/A") else form


def _is_material(form: str) -> bool:
    base = _base_form(form)
    return base in _FORM_DESC or base.startswith("424B")


def _describe(r: dict) -> str:
    form = r["form"]
    base = _base_form(form)
    if base.startswith("424B"):
        desc = "Prospectus supplement"
    else:
        desc = _FORM_DESC.get(base, "Filing")
    if form.endswith("/A"):
        desc += " (amendment)"
    if base in ("8-K", "6-K") and r["items"]:
        codes = [c.strip() for c in str(r["items"]).split(",") if c.strip()]
        if len(codes) > 1:
            codes = [c for c in codes if c != "9.01"]  # exhibits-only item adds nothing
        if codes:
            desc = "Items: " + "; ".join(f"{c} {ITEM_8K.get(c, '')}".strip() for c in codes)
    elif r["primaryDocDescription"]:
        extra = str(r["primaryDocDescription"]).strip()
        trivial = re.sub(r"[^A-Z0-9]", "", extra.upper().replace("FORM", ""))
        if trivial and trivial != re.sub(r"[^A-Z0-9]", "", base.upper()):
            desc += f": {extra[:80]}"
    return desc


def _doc_url(cik: int, r: dict) -> str:
    acc = str(r["accessionNumber"]).replace("-", "")
    doc = r["primaryDocument"]
    if not doc:
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
    return ARCHIVE_URL.format(cik=cik, accession=acc, doc=doc)


# --------------------------------------------------------------------------- HTML cleaning

_DROP_BLOCKS = re.compile(
    r"<(script|style|head|title|ix:header|xbrl)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HIDDEN_DIV = re.compile(r"<div[^>]*display\s*:\s*none[^>]*>.*?</div>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_TAG = re.compile(r"<\s*(br|/p|/div|/tr|/li|/h[1-6]|/table)\b[^>]*>", re.IGNORECASE)
_CELL_TAG = re.compile(r"<\s*/t[dh]\s*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_ITEM_HEAD_LINE = re.compile(r"^Item\s*[1-9]\.\d{2}\b", re.IGNORECASE | re.MULTILINE)
_ITEM_HEAD = re.compile(r"\bItem\s*[1-9]\.\d{2}\b", re.IGNORECASE)
# Tail headings must stand on their own line, so "signature" inside a sentence is kept.
_TAIL = re.compile(r"^(?:SIGNATURES?|EXHIBIT\s+INDEX|Item\s*9\.01\b.*)$", re.IGNORECASE | re.MULTILINE)


def _clean_html(raw: str) -> str:
    """Visible text of an EDGAR HTML/text document, whitespace-normalised by line."""
    text = _COMMENT.sub(" ", raw)
    text = _DROP_BLOCKS.sub(" ", text)
    text = _HIDDEN_DIV.sub(" ", text)
    text = _BLOCK_TAG.sub("\n", text)
    text = _CELL_TAG.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text).replace("\xa0", " ").replace("​", "")
    lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _strip_8k_boilerplate(text: str) -> str:
    """Drop the 8-K cover page (before the first Item heading) and the signatures/exhibits tail."""
    first = _ITEM_HEAD_LINE.search(text) or _ITEM_HEAD.search(text)
    if not first:
        return text
    body = text[first.start():]
    # Skip past the first heading so a leading "Item 9.01" is not cut immediately.
    tail = _TAIL.search(body, len(first.group(0)))
    return body[: tail.start()] if tail else body


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.7:
        cut = cut[:space]
    return cut.rstrip() + " ..."


def _excerpt(url: str) -> str:
    try:
        resp = _http_get(url)
    except _SecError as exc:
        return f"(excerpt unavailable: {exc})"
    body = _strip_8k_boilerplate(_clean_html(resp.text))
    return _truncate(re.sub(r"\s+", " ", body).strip(), EXCERPT_CHARS) or "(empty document)"


# --------------------------------------------------------------------------- public API


def _parse_day(value: str) -> Optional[str]:
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date().isoformat()
    except ValueError:
        return None


def _resolve(ticker: str) -> Tuple[Optional[dict], Optional[str]]:
    """(entry, None) or (None, short explanation)."""
    t = (ticker or "").strip().upper()
    if market_of(t) != "US":
        return None, f"SEC EDGAR covers US-listed companies only; {t or ticker!r} is not a US ticker."
    if not _user_agent():
        return None, NOT_CONFIGURED
    try:
        entry = _lookup(t)
    except _SecError as exc:
        return None, str(exc)
    if not entry:
        return None, (f"SEC EDGAR: {t} is not in the SEC company ticker list "
                      "(ETFs and some funds are not listed); no filings available.")
    return entry, None


def get_filings(ticker: str, start_date: str, end_date: str, limit: int = 20,
                excerpts: int = 2) -> str:
    """Material SEC filings for ``ticker`` filed within [start_date, end_date], as Markdown.

    end_date is the analysis date: nothing filed after it is included. Form 3/4/5 and
    other routine forms are summarised as counts. The newest ``excerpts`` 8-K filings
    get a ~600-char excerpt of their item text.
    """
    try:
        return _get_filings(ticker, start_date, end_date, limit, excerpts)
    except _SecError as exc:
        return str(exc)
    except Exception as exc:  # never break an analyst run over this source
        return f"SEC EDGAR error: {type(exc).__name__}."


def _get_filings(ticker, start_date, end_date, limit, excerpts) -> str:
    start, end = _parse_day(start_date), _parse_day(end_date)
    if not start or not end:
        return "SEC EDGAR: dates must be YYYY-MM-DD."
    if start > end:
        start, end = end, start
    entry, problem = _resolve(ticker)
    if problem:
        return problem
    cik = entry["cik"]
    rows, note = _filings_in_range(cik, start, end)

    lines = [f"## SEC EDGAR filings: {entry['ticker']} ({entry['title']}, {start} ~ {end})"]
    material, insider, other = [], [], {}
    prospectus_listed = 0
    for r in rows:
        form = r["form"]
        base = _base_form(form)
        if base in _INSIDER_FORMS:
            insider.append(r)
        elif _is_material(form):
            if base.startswith("424B"):
                if prospectus_listed >= _PROSPECTUS_CAP:
                    other.setdefault(form, []).append(r)
                    continue
                prospectus_listed += 1
            material.append(r)
        else:
            other.setdefault(form, []).append(r)

    limit = max(int(limit or 0), 0)
    shown = material[:limit]
    excerpts_left = max(int(excerpts or 0), 0)
    if not shown:
        lines.append("No material filings (10-K/10-Q/8-K/6-K/registrations/13D/13G/proxy) in range.")
    for r in shown:
        period = ""
        if _base_form(r["form"]) in ("10-K", "10-Q", "20-F", "40-F") and r["reportDate"]:
            period = f" (period {r['reportDate']})"
        url = _doc_url(cik, r)
        lines.append(f"- {r['filingDate']} | {r['form']}{period} | {_describe(r)} | {url}")
        if excerpts_left and _base_form(r["form"]) == "8-K" and r["primaryDocument"]:
            excerpts_left -= 1
            lines.append(f"  > {_excerpt(url)}")
    if len(material) > len(shown):
        lines.append(f"- ... {len(material) - len(shown)} more material filings in range not listed.")

    if insider:
        lines.append(f"Form 3/4/5 insider filings: {len(insider)} in range, "
                     f"latest {insider[0]['filingDate']}.")
    if other:
        counts = sorted(other.items(), key=lambda kv: -len(kv[1]))
        latest = max(r["filingDate"] for group in other.values() for r in group)
        summary = ", ".join(f"{form} x{len(group)}" for form, group in counts[:8])
        if len(counts) > 8:
            summary += ", ..."
        lines.append(f"Other filings: {summary} (latest {latest}).")
    if note:
        lines.append(note)
    lines.append(f"Source: SEC EDGAR, filed on or before {end}.")
    return "\n".join(lines)


def get_filing_text(url: str, max_chars: int = 4000) -> str:
    """Cleaned text of one EDGAR document (https://www.sec.gov/ URLs only)."""
    try:
        parsed = urlparse(str(url or "").strip())
    except ValueError:
        parsed = None
    if not parsed or parsed.scheme != "https" or parsed.netloc.lower() != "www.sec.gov":
        return "Only https://www.sec.gov/ document URLs are allowed."
    try:
        resp = _http_get(parsed.geturl())
    except _SecError as exc:
        return str(exc)
    except Exception as exc:
        return f"SEC EDGAR error: {type(exc).__name__}."
    ctype = (resp.headers.get("Content-Type") or "").lower()
    if ctype and not any(k in ctype for k in ("html", "xml", "text", "json")):
        return f"Unsupported SEC document type ({ctype.split(';')[0]}); open it in a browser: {url}"
    text = _clean_html(resp.text)
    if re.search(r"\bFORM\s+8-K\b", text[:3000], re.IGNORECASE):
        text = _strip_8k_boilerplate(text)
    text = _truncate(text, max(int(max_chars or 0), 200))
    return f"### SEC document: {url}\n{text}" if text else f"SEC document is empty: {url}"
