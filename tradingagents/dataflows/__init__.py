"""Data vendors. yfinance is used by several modules here, so its cache location is set once on import."""
from tradingagents.default_config import configure_yfinance_cache

# yfinance 캐시(시간대·쿠키)를 사용자 AppData 대신 저장소 data/cache/yfinance 에 둔다.
configure_yfinance_cache()
