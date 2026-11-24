"""OpenDART (금융감독원 전자공시) client: disclosure list and key financial accounts.

Public functions return compact Markdown for the analysts and never raise: a non-Korean
ticker, a missing ``DART_API_KEY``, DART errors and network failures all come back as a
short explanatory string. The API key is never logged or echoed (request exceptions
carry the full URL, so their text is scrubbed).

The stock-code -> corp_code mapping (corpCode.xml, a ZIP of ~100k companies) is
downloaded at most once every 7 days, cached as JSON in data/cache/dart and kept in
memory. Everything is thread-safe.
"""

import io
import json
import os
import re
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime
from typing import Optional

import requests

from .external_common import RateLimiter, cache_dir, get_env, korean_code

API_BASE = "https://opendart.fss.or.kr/api"
VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={}"

_TIMEOUT = 15
_RETRY_DELAY = 1.0
_LIMITER = RateLimiter(0.15)

_CORP_FILE = "corp_codes.json"
_CORP_MAX_AGE = 7 * 24 * 3600
_CORP_RETRY_AFTER = 3600  # after a failed download, wait this long before trying again

_NOT_CONFIGURED = "DART is not configured: set DART_API_KEY in .env to enable Korean disclosures."

_STATUS_MESSAGES = {
    "010": "DART API key is not registered (status 010). Check DART_API_KEY.",
    "011": "DART API key is not usable (status 011). Check DART_API_KEY.",
    "012": "DART API key is not allowed from this IP (status 012).",
    "020": "DART request limit exceeded (status 020); try again later.",
    "021": "DART request covers too many companies (status 021).",
    "100": "DART rejected a request field (status 100).",
    "101": "DART rejected the request (status 101).",
    "800": "DART is under maintenance (status 800); try again later.",
    "900": "DART reported an undefined error (status 900).",
    "901": "DART API key has expired (status 901). Renew DART_API_KEY.",
}

_KEY_PARAM = re.compile(r"(crtfc_key=)[^&\s'\"]+", re.IGNORECASE)


class _DartError(Exception):
    """A short, key-free message ready to hand back to the caller."""


class _NoData(Exception):
    """DART status 013: the query matched nothing."""


def _scrub(text: str, key: str = "") -> str:
    text = _KEY_PARAM.sub(r"\1***", text or "")
    if key:
        text = text.replace(key, "***")
    return text[:200]


def _http_get(endpoint: str, params: dict, key: str) -> requests.Response:
    """GET with rate limiting, a timeout and one retry on timeouts / connection errors / 5xx."""
    url = f"{API_BASE}/{endpoint}"
    query = {"crtfc_key": key, **params}
    for attempt in (1, 2):
        _LIMITER.wait()
        try:
            resp = requests.get(url, params=query, timeout=_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 1:
                time.sleep(_RETRY_DELAY)
                continue
            raise _DartError(f"DART request failed ({type(exc).__name__}): {_scrub(str(exc), key)}")
        except requests.RequestException as exc:
            raise _DartError(f"DART request failed ({type(exc).__name__}): {_scrub(str(exc), key)}")
        if resp.status_code >= 500 and attempt == 1:
            time.sleep(_RETRY_DELAY)
            continue
        if resp.status_code != 200:
            raise _DartError(f"DART returned HTTP {resp.status_code} for {endpoint}.")
        return resp
    raise _DartError(f"DART request failed for {endpoint}.")  # pragma: no cover


def _check_status(status: str, message: str = "") -> None:
    status = (status or "").strip()
    if status == "000":
        return
    if status == "013":
        raise _NoData()
    text = _STATUS_MESSAGES.get(status) or f"DART error (status {status or '?'}): {message[:100]}"
    raise _DartError(text)


def _get_json(endpoint: str, params: dict, key: str) -> dict:
    resp = _http_get(endpoint, params, key)
    try:
        data = resp.json()
    except ValueError:
        raise _DartError(f"DART returned an unreadable response for {endpoint}.")
    if not isinstance(data, dict):
        raise _DartError(f"DART returned an unexpected response for {endpoint}.")
    _check_status(str(data.get("status", "")), str(data.get("message", "")))
    return data


# --------------------------------------------------------------------------- corp codes


class _CorpCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.corps: Optional[dict] = None  # stock_code -> {"corp_code", "corp_name"}
        self.fetched_at = 0.0
        self.failed_at = 0.0

    def fresh(self, now: float) -> bool:
        return self.corps is not None and now - self.fetched_at < _CORP_MAX_AGE


_CORP = _CorpCache()


def _parse_corp_zip(content: bytes) -> dict:
    """{stock_code: {"corp_code", "corp_name"}} for listed companies in a corpCode.xml ZIP."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not names:
            raise _DartError("DART corpCode.xml archive has no XML file.")
        xml_bytes = zf.read(names[0])
    corps = {}
    for node in ET.fromstring(xml_bytes).iter("list"):
        stock = (node.findtext("stock_code") or "").strip()
        if not stock:
            continue
        corps[stock] = {
            "corp_code": (node.findtext("corp_code") or "").strip(),
            "corp_name": (node.findtext("corp_name") or "").strip(),
        }
    if not corps:
        raise _DartError("DART corpCode.xml contained no listed companies.")
    return corps


def _download_corps(key: str) -> dict:
    resp = _http_get("corpCode.xml", {}, key)
    content = resp.content or b""
    if not zipfile.is_zipfile(io.BytesIO(content)):
        # Errors come back as a small XML document instead of the ZIP.
        text = content[:2000].decode("utf-8", "replace")
        status = re.search(r"<status>\s*(\d+)\s*</status>", text)
        message = re.search(r"<message>(.*?)</message>", text, re.S)
        _check_status(status.group(1) if status else "", message.group(1) if message else "")
        raise _DartError("DART corpCode.xml download was not a ZIP archive.")
    try:
        return _parse_corp_zip(content)
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise _DartError(f"DART corpCode.xml could not be parsed ({type(exc).__name__}).")


def _read_corp_file(path) -> Optional[tuple]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        corps = payload["corps"]
        fetched_at = float(payload["fetched_at"])
        if isinstance(corps, dict) and corps:
            return corps, fetched_at
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _write_corp_file(path, corps: dict, fetched_at: float) -> None:
    payload = {"fetched_at": fetched_at, "corps": corps}
    fd, tmp = tempfile.mkstemp(prefix=".corp_codes-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _corp_map(key: str) -> dict:
    """The stock_code mapping, from memory, the JSON cache or (at most weekly) DART."""
    now = time.time()
    if _CORP.fresh(now):
        return _CORP.corps
    with _CORP.lock:
        now = time.time()
        if _CORP.fresh(now):
            return _CORP.corps
        path = cache_dir("dart") / _CORP_FILE
        if _CORP.corps is None:
            cached = _read_corp_file(path)
            if cached:
                _CORP.corps, _CORP.fetched_at = cached
                if _CORP.fresh(now):
                    return _CORP.corps
        # Missing or stale: download, unless a recent attempt failed and a stale copy exists.
        if _CORP.corps is not None and now - _CORP.failed_at < _CORP_RETRY_AFTER:
            return _CORP.corps
        try:
            corps = _download_corps(key)
        except (_DartError, _NoData) as exc:
            _CORP.failed_at = now
            if _CORP.corps is not None:
                return _CORP.corps  # stale beats nothing
            if isinstance(exc, _NoData):
                raise _DartError("DART corpCode.xml returned no data.")
            raise
        _CORP.corps, _CORP.fetched_at, _CORP.failed_at = corps, now, 0.0
        _write_corp_file(path, corps, now)
        return corps


def _lookup(ticker: str):
    """(info, None) or (None, short message)."""
    code = korean_code(ticker)
    if not code:
        return None, f"DART covers Korean listings only; '{ticker}' is not a Korean ticker (e.g. 005930.KS)."
    key = get_env("DART_API_KEY")
    if not key:
        return None, _NOT_CONFIGURED
    try:
        corps = _corp_map(key)
    except _DartError as exc:
        return None, str(exc)
    row = corps.get(code)
    if not row or not row.get("corp_code"):
        return None, f"DART has no listed company with stock code {code}."
    return {"corp_code": row["corp_code"], "corp_name": row.get("corp_name", ""), "stock_code": code}, None


def get_corp_info(ticker: str) -> Optional[dict]:
    """{"corp_code", "corp_name", "stock_code"} for a listed Korean company, else None."""
    info, _ = _lookup(ticker)
    return info


# --------------------------------------------------------------------------- helpers


def _ymd(value: str) -> Optional[str]:
    """'2026-11-22' / '20261122' -> '20261122', else None."""
    digits = re.sub(r"[^0-9]", "", str(value or ""))[:8]
    if len(digits) != 8:
        return None
    try:
        datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return None
    return digits


def _iso(ymd: str) -> str:
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}" if len(ymd) == 8 else ymd


# --------------------------------------------------------------------------- disclosures

_IMPORTANT = re.compile(
    "|".join([
        r"단일판매", r"공급계약", r"유상증자", r"무상증자", r"감자", r"영업\(잠정\)실적", r"잠정실적",
        r"매출액또는손익구조", r"주요사항보고서", r"자기주식", r"전환사채", r"신주인수권부사채",
        r"교환사채", r"합병", r"분할", r"최대주주", r"소송", r"영업양수", r"영업양도", r"영업정지",
        r"공개매수", r"부도", r"회생", r"상장폐지", r"관리종목", r"횡령", r"배임", r"불성실공시",
        r"거래정지", r"매매거래정지", r"조회공시", r"현금.?현물배당", r"타법인주식", r"유형자산\s*(취득|처분)",
        r"투자판단관련주요경영사항", r"채무보증", r"자산양수도",
    ])
)


def _is_important(report_nm: str) -> bool:
    return bool(_IMPORTANT.search(re.sub(r"\s+", "", report_nm or "")))


def get_disclosures(ticker: str, start_date: str, end_date: str, limit: int = 20) -> str:
    """Recent DART filings in [start_date, end_date], newest first, important ones (★) first."""
    info, err = _lookup(ticker)
    if err:
        return err
    bgn, end = _ymd(start_date), _ymd(end_date)
    if not bgn or not end:
        return f"Invalid date range for DART disclosures: {start_date!r} to {end_date!r} (use YYYY-MM-DD)."
    if bgn > end:
        bgn, end = end, bgn
    limit = max(1, min(int(limit or 20), 200))
    key = get_env("DART_API_KEY")
    header = (f"## DART 공시: {info['corp_name']} ({info['stock_code']}, corp {info['corp_code']}) "
              f"{_iso(bgn)} ~ {_iso(end)}")

    # Fetch at least one full page so important filings are not crowded out by routine ones.
    wanted = max(limit, 100)
    rows, total, page = [], 0, 1
    try:
        while len(rows) < wanted:
            data = _get_json("list.json", {
                "corp_code": info["corp_code"], "bgn_de": bgn, "end_de": end,
                "page_no": page, "page_count": 100, "sort": "date", "sort_mth": "desc",
            }, key)
            batch = data.get("list") or []
            rows.extend(batch)
            total = int(data.get("total_count") or len(rows))
            if not batch or page >= int(data.get("total_page") or 1):
                break
            page += 1
    except _NoData:
        return f"{header}\nNo disclosures in range."
    except _DartError as exc:
        return f"{header}\n{exc}"

    seen, items = set(), []
    for r in rows:
        dt = str(r.get("rcept_dt") or "")
        no = str(r.get("rcept_no") or "")
        if not dt or dt > end or dt < bgn or no in seen:  # no look-ahead past the analysis date
            continue
        seen.add(no)
        items.append(r)
    if not items:
        return f"{header}\nNo disclosures in range."
    items.sort(key=lambda r: (str(r.get("rcept_dt")), str(r.get("rcept_no"))), reverse=True)

    important = [r for r in items if _is_important(r.get("report_nm", ""))]
    routine = [r for r in items if not _is_important(r.get("report_nm", ""))]
    shown = (important + routine)[:limit]

    def line(r) -> str:
        star = "★ " if _is_important(r.get("report_nm", "")) else ""
        name = re.sub(r"\s+", " ", str(r.get("report_nm") or "")).strip()
        filer = str(r.get("flr_nm") or "").strip()
        return f"- {star}{_iso(str(r['rcept_dt']))} {name} — {filer} {VIEWER_URL.format(r['rcept_no'])}"

    out = [header, f"총 {max(total, len(items))}건 중 {len(shown)}건 표시 (★ 주요 공시 {len(important)}건 우선, 최신순)"]
    out.extend(line(r) for r in shown)
    hidden = len(items) - len(shown)
    if hidden > 0:
        out.append(f"(+{hidden}건 생략)")
    return "\n".join(out)


# --------------------------------------------------------------------------- financials

_REPORT_NAMES = {"11013": "1분기보고서", "11012": "반기보고서", "11014": "3분기보고서", "11011": "사업보고서"}
_PERIOD_END = {"11013": (3, 31), "11012": (6, 30), "11014": (9, 30), "11011": (12, 31)}

# (label, statement, matcher on the space-free account name)
_ACCOUNTS = [
    ("매출액", "IS", lambda n: n in ("매출액", "수익(매출액)", "영업수익", "매출", "수익") or n.startswith("매출액")),
    ("영업이익", "IS", lambda n: n.startswith("영업이익")),
    ("당기순이익", "IS", lambda n: n.startswith("당기순이익") or n.startswith("분기순이익") or n.startswith("반기순이익")),
    ("자산총계", "BS", lambda n: n == "자산총계"),
    ("부채총계", "BS", lambda n: n == "부채총계"),
    ("자본총계", "BS", lambda n: n == "자본총계"),
]


def _amount(value) -> Optional[int]:
    text = str(value or "").replace(",", "").strip()
    if text in ("", "-"):
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _fmt_krw(v: Optional[int]) -> str:
    if v is None:
        return "-"
    if abs(v) >= 10**12:
        return f"{v / 10**12:,.2f}조"
    if abs(v) >= 10**8:
        return f"{v / 10**8:,.0f}억"
    return f"{v:,}"


def _change(cur: Optional[int], prev: Optional[int]) -> str:
    if cur is None or prev is None:
        return "-"
    if prev < 0 <= cur:
        return "흑자전환"
    if cur < 0 <= prev and prev != 0:
        return "적자전환"
    if cur < 0 and prev < 0:
        return "적자지속"
    if prev == 0:
        return "-"
    return f"{(cur - prev) / abs(prev) * 100:+.1f}%"


def _candidates(curr: str):
    """(year, reprt_code) newest first, only for periods that ended before curr_date."""
    today = datetime.strptime(curr, "%Y%m%d").date()
    out = []
    for year in (today.year, today.year - 1, today.year - 2):
        for code in ("11011", "11014", "11012", "11013"):
            m, d = _PERIOD_END[code]
            if date(year, m, d) < today:
                out.append((year, code))
    return out


def _pick_rows(rows: list) -> tuple:
    """Prefer consolidated (CFS) rows; fall back to separate (OFS)."""
    for fs in ("CFS", "OFS"):
        chosen = [r for r in rows if r.get("fs_div") == fs]
        if chosen:
            return fs, chosen
    return "", rows


def _report_table(rows: list, reprt_code: str) -> list:
    fs, rows = _pick_rows(rows)
    quarterly = reprt_code != "11011"
    currency = next((r.get("currency") for r in rows if r.get("currency")), "KRW")
    is_lines, cumulative, bs_lines = [], [], []
    for label, sj, match in _ACCOUNTS:
        row = next((r for r in rows if r.get("sj_div") == sj
                    and match(re.sub(r"\s+", "", str(r.get("account_nm") or "")))), None)
        if not row:
            continue
        cur, prev = _amount(row.get("thstrm_amount")), _amount(row.get("frmtrm_amount"))
        line = f"| {label} | {_fmt_krw(cur)} | {_fmt_krw(prev)} | {_change(cur, prev)} |"
        (is_lines if sj == "IS" else bs_lines).append(line)
        if quarterly and sj == "IS":
            c_cur, c_prev = _amount(row.get("thstrm_add_amount")), _amount(row.get("frmtrm_add_amount"))
            if c_cur is not None and c_cur != cur:
                cumulative.append(f"| {label}(누적) | {_fmt_krw(c_cur)} | {_fmt_krw(c_prev)} | {_change(c_cur, c_prev)} |")
    if not (is_lines or bs_lines):
        return []
    lines = ["| 계정 | 당기 | 비교기 | 증감 |", "|---|---|---|---|"] + is_lines + cumulative + bs_lines
    fs_label = {"CFS": "연결", "OFS": "별도"}.get(fs, fs or "?")
    if quarterly:
        basis = "손익: 3개월 vs 전년 동기, 재무상태: 분기말 vs 전기말"
    else:
        basis = "손익·재무상태: 당기 vs 전기"
    unit = "원(조/억)" if currency == "KRW" else currency
    return [f"{fs_label}재무제표 · {unit} · {basis}"] + lines


def get_financial_summary(ticker: str, curr_date: str) -> str:
    """Key accounts from the latest periodic report(s) filed on or before curr_date."""
    info, err = _lookup(ticker)
    if err:
        return err
    curr = _ymd(curr_date)
    if not curr:
        return f"Invalid date for DART financial summary: {curr_date!r} (use YYYY-MM-DD)."
    key = get_env("DART_API_KEY")
    header = (f"## DART 재무 요약: {info['corp_name']} ({info['stock_code']}, corp {info['corp_code']}) "
              f"— {_iso(curr)} 기준 공시분")

    sections, attempts = [], 0
    for year, code in _candidates(curr):
        if len(sections) >= 2 or attempts >= 6:
            break
        # After the latest report, only add the latest annual report as context.
        if sections and code != "11011":
            continue
        attempts += 1
        try:
            data = _get_json("fnlttSinglAcnt.json", {
                "corp_code": info["corp_code"], "bsns_year": str(year), "reprt_code": code,
            }, key)
        except _NoData:
            continue
        except _DartError as exc:
            if sections:
                break
            return f"{header}\n{exc}"
        rows = data.get("list") or []
        rcept_no = next((str(r.get("rcept_no")) for r in rows if r.get("rcept_no")), "")
        filed = rcept_no[:8]
        if not filed or filed > curr:  # filed after the analysis date: not yet public
            continue
        table = _report_table(rows, code)
        if not table:
            continue
        title = (f"### {year} {_REPORT_NAMES[code]} (공시 {_iso(filed)}) "
                 f"{VIEWER_URL.format(rcept_no)}")
        sections.append("\n".join([title] + table))
        if code == "11011":
            break
    if not sections:
        return f"{header}\nNo periodic financial report filed on or before {_iso(curr)} was found."
    return "\n\n".join([header] + sections)
