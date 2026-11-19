"""종목 티커 자동완성용 목록과 검색 (app.py 전용).

- POPULAR: 자주 쓰는 종목. 한글 이름을 붙여 두어 드롭다운에서 '엔비디아', '삼성'처럼
  한글로 쳐도 걸러진다.
- search(): Yahoo Finance 검색. 목록에 없는 종목을 영문 이름이나 티커 일부로 찾는다.
  Yahoo 검색은 한글 이름을 지원하지 않는다.
"""
import re
from typing import Dict, Iterable, List, Tuple

POPULAR: List[Tuple[str, str]] = [
    # 미국 빅테크·주요 종목
    ("NVDA", "NVIDIA · 엔비디아"),
    ("AAPL", "Apple · 애플"),
    ("MSFT", "Microsoft · 마이크로소프트"),
    ("GOOGL", "Alphabet A · 구글"),
    ("AMZN", "Amazon · 아마존"),
    ("META", "Meta Platforms · 메타"),
    ("TSLA", "Tesla · 테슬라"),
    ("AVGO", "Broadcom · 브로드컴"),
    ("AMD", "AMD · 에이엠디"),
    ("INTC", "Intel · 인텔"),
    ("TSM", "TSMC · 대만 반도체"),
    ("MU", "Micron · 마이크론"),
    ("QCOM", "Qualcomm · 퀄컴"),
    ("ARM", "Arm Holdings · 암"),
    ("PLTR", "Palantir · 팔란티어"),
    ("NFLX", "Netflix · 넷플릭스"),
    ("ORCL", "Oracle · 오라클"),
    ("CRM", "Salesforce · 세일즈포스"),
    ("ADBE", "Adobe · 어도비"),
    ("IONQ", "IonQ · 아이온큐"),
    ("SMCI", "Super Micro Computer · 슈퍼마이크로"),
    ("COIN", "Coinbase · 코인베이스"),
    ("MSTR", "MicroStrategy · 마이크로스트래티지"),
    ("BRK-B", "Berkshire Hathaway B · 버크셔 해서웨이"),
    ("JPM", "JPMorgan Chase · JP모건"),
    ("V", "Visa · 비자"),
    ("MA", "Mastercard · 마스터카드"),
    ("LLY", "Eli Lilly · 일라이 릴리"),
    ("NVO", "Novo Nordisk · 노보 노디스크"),
    ("UNH", "UnitedHealth · 유나이티드헬스"),
    ("JNJ", "Johnson & Johnson · 존슨앤드존슨"),
    ("XOM", "Exxon Mobil · 엑슨모빌"),
    ("KO", "Coca-Cola · 코카콜라"),
    ("WMT", "Walmart · 월마트"),
    ("COST", "Costco · 코스트코"),
    ("DIS", "Disney · 디즈니"),
    ("NKE", "Nike · 나이키"),
    ("BA", "Boeing · 보잉"),
    # ETF (레버리지·인버스 포함)
    ("SPY", "SPDR S&P 500 ETF"),
    ("VOO", "Vanguard S&P 500 ETF"),
    ("QQQ", "Invesco QQQ (나스닥 100)"),
    ("DIA", "SPDR 다우존스 ETF"),
    ("IWM", "iShares 러셀 2000 ETF"),
    ("SCHD", "Schwab 미국 배당 ETF"),
    ("SOXX", "iShares 반도체 ETF"),
    ("SMH", "VanEck 반도체 ETF"),
    ("TQQQ", "ProShares 나스닥 100 3배"),
    ("SQQQ", "ProShares 나스닥 100 인버스 3배"),
    ("SOXL", "Direxion 반도체 3배"),
    ("SOXS", "Direxion 반도체 인버스 3배"),
    ("UPRO", "ProShares S&P 500 3배"),
    ("TMF", "Direxion 미국 장기채 3배"),
    ("TLT", "iShares 미국 장기채 20년+"),
    ("GLD", "SPDR 금 ETF"),
    ("SLV", "iShares 은 ETF"),
    ("USO", "United States Oil Fund (원유)"),
    ("SCO", "ProShares 원유 인버스 2배"),
    ("EWY", "iShares MSCI 한국 ETF"),
    ("KORU", "Direxion 한국 3배"),
    ("IBIT", "iShares 비트코인 현물 ETF"),
    ("ARKK", "ARK Innovation ETF"),
    # 지수
    ("^GSPC", "S&P 500 지수"),
    ("^IXIC", "나스닥 종합 지수"),
    ("^DJI", "다우존스 지수"),
    ("^KS11", "코스피 지수"),
    ("^KQ11", "코스닥 지수"),
    # 한국 (.KS 코스피, .KQ 코스닥)
    ("005930.KS", "삼성전자 · Samsung Electronics"),
    ("000660.KS", "SK하이닉스 · SK Hynix"),
    ("373220.KS", "LG에너지솔루션 · LG Energy Solution"),
    ("207940.KS", "삼성바이오로직스 · Samsung Biologics"),
    ("005380.KS", "현대차 · Hyundai Motor"),
    ("000270.KS", "기아 · Kia"),
    ("035420.KS", "NAVER · 네이버"),
    ("035720.KS", "카카오 · Kakao"),
    ("005490.KS", "POSCO홀딩스 · 포스코"),
    ("051910.KS", "LG화학 · LG Chem"),
    ("006400.KS", "삼성SDI · Samsung SDI"),
    ("068270.KS", "셀트리온 · Celltrion"),
    ("105560.KS", "KB금융 · KB Financial"),
    ("055550.KS", "신한지주 · Shinhan Financial"),
    ("012330.KS", "현대모비스 · Hyundai Mobis"),
    ("066570.KS", "LG전자 · LG Electronics"),
    ("034020.KS", "두산에너빌리티 · Doosan Enerbility"),
    ("012450.KS", "한화에어로스페이스 · Hanwha Aerospace"),
    ("042660.KS", "한화오션 · Hanwha Ocean"),
    ("329180.KS", "HD현대중공업 · HD Hyundai Heavy"),
    ("086520.KQ", "에코프로 · EcoPro"),
    ("247540.KQ", "에코프로비엠 · EcoPro BM"),
    ("196170.KQ", "알테오젠 · Alteogen"),
    # 암호화폐
    ("BTC-USD", "비트코인 · Bitcoin"),
    ("ETH-USD", "이더리움 · Ethereum"),
    ("SOL-USD", "솔라나 · Solana"),
    ("XRP-USD", "리플 · XRP"),
    ("DOGE-USD", "도지코인 · Dogecoin"),
    ("ADA-USD", "에이다 · Cardano"),
]

# 분석할 수 있는 종류만 검색 결과에 남긴다 (선물·옵션 제외).
_SEARCH_TYPES = {
    "EQUITY": "주식",
    "ETF": "ETF",
    "CRYPTOCURRENCY": "암호화폐",
    "INDEX": "지수",
    "MUTUALFUND": "펀드",
    "CURRENCY": "환율",
}

_TICKER_RE = re.compile(r"^[A-Z0-9^][A-Z0-9.\-=^]{0,19}$")


def normalize(symbol: str) -> str:
    return symbol.strip().upper()


def is_valid_ticker(symbol: str) -> bool:
    """yfinance 티커 모양인지 (한글·공백 등 회사명을 그대로 친 경우를 거른다)."""
    return bool(_TICKER_RE.match(normalize(symbol)))


def search(query: str, limit: int = 10) -> List[Tuple[str, str]]:
    """Yahoo Finance에서 이름·티커로 종목을 찾아 (티커, 설명) 목록을 반환한다. 실패하면 []."""
    query = query.strip()
    if not query:
        return []
    try:
        import yfinance as yf

        quotes = yf.Search(query, max_results=limit, news_count=0).quotes
    except Exception:
        return []
    results = []
    for q in quotes:
        symbol, kind = q.get("symbol"), _SEARCH_TYPES.get(q.get("quoteType", ""))
        if not symbol or not kind:
            continue
        name = q.get("shortname") or q.get("longname") or ""
        exchange = q.get("exchDisp") or ""
        results.append((symbol, " · ".join(p for p in (name.strip(), kind, exchange) if p)))
    return results


def merge(*groups: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    """여러 (티커, 설명) 목록을 순서대로 합친다. 같은 티커는 먼저 나온 설명을 쓴다."""
    merged: Dict[str, str] = {}
    for group in groups:
        for symbol, label in group:
            merged.setdefault(symbol, label)
    return merged
