from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor

MAX_INDICATORS = 12
MAX_LOOK_BACK_DAYS = 365

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "one indicator name, or several comma-separated names (e.g. 'rsi,macd,boll')"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve one or more technical indicators for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): An indicator name such as 'rsi', or a comma-separated list such as
            'close_50_sma,macd,rsi,boll'. Request all the indicators you need in one call.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    # Bound the work a single call can trigger (model- or prompt-injection-driven).
    indicators = list(dict.fromkeys(i.strip().lower() for i in indicator.split(",") if i.strip()))[:MAX_INDICATORS]
    look_back_days = max(1, min(int(look_back_days), MAX_LOOK_BACK_DAYS))
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)