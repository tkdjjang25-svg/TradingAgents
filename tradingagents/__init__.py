"""TradingAgents package."""
from tradingagents.default_config import configure_yfinance_cache

# yfinance 캐시가 사용자 AppData 대신 저장소 data/cache/yfinance 에 생기도록.
configure_yfinance_cache()
