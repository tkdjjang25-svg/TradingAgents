"""yfinance-based news data fetching functions."""

import re

import yfinance as yf
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from .stockstats_utils import yf_retry


def _extract_article_data(article: dict) -> dict:
    """Extract article data from yfinance news format (handles nested 'content' structure)."""
    # Handle nested content structure
    if "content" in article:
        content = article["content"]
        title = content.get("title", "No title")
        summary = content.get("summary", "")
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "Unknown")

        # Get URL from canonicalUrl or clickThroughUrl
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = url_obj.get("url", "")

        # Get publish date
        pub_date_str = content.get("pubDate", "")
        pub_date = None
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass

        return {
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "link": link,
            "pub_date": pub_date,
        }
    else:
        # Flat structure (yf.Search news): publish time is a unix timestamp
        pub_date = None
        ts = article.get("providerPublishTime")
        if ts:
            try:
                pub_date = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except (ValueError, OverflowError, OSError, TypeError):
                pass
        return {
            "title": article.get("title", "No title"),
            "summary": article.get("summary", ""),
            "publisher": article.get("publisher", "Unknown"),
            "link": article.get("link", ""),
            "pub_date": pub_date,
        }


_CORP_SUFFIX = re.compile(
    r"[\s,]+(co\.?,?\s*ltd\.?|company,?\s*limited|corporation|corp\.?|inc\.?|ltd\.?|plc|"
    r"limited|holdings?|group|s\.?a\.?|n\.?v\.?|ag|se)$",
    re.IGNORECASE,
)


def _company_query(name: str) -> str:
    """'Samsung Electronics Co., Ltd.' -> 'Samsung Electronics'.

    Yahoo's news search matches loosely on legal suffixes and returns unrelated
    articles, while the bare company name finds the company's own news.
    """
    name = name.strip()
    while True:
        shorter = _CORP_SUFFIX.sub("", name).strip(" ,")
        if shorter == name or not shorter:
            return name
        name = shorter


def _search_news(ticker: str, count: int = 20) -> list:
    """News via yf.Search, used when Ticker.get_news fails.

    Ticker.get_news hits Yahoo's ncp endpoint, which sometimes answers with a
    "Will be right back" page that yfinance reports as "Yahoo! Finance is
    currently down", while the search endpoint keeps working. Searching by the
    raw symbol finds nothing for many non-US tickers (e.g. 005930.KS), so the
    company name from the search quotes is tried as well.
    """
    search = yf_retry(lambda: yf.Search(ticker, max_results=5, news_count=count))
    if search.news:
        return search.news
    for quote in search.quotes or []:
        if str(quote.get("symbol", "")).upper() != ticker.upper():
            continue
        name = _company_query(quote.get("longname") or quote.get("shortname") or "")
        if name:
            return yf_retry(lambda: yf.Search(name, max_results=0, news_count=count)).news or []
    return []


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        Formatted string containing news articles
    """
    try:
        stock = yf.Ticker(ticker)
        try:
            news = yf_retry(lambda: stock.get_news(count=20))
        except Exception:
            news = None
        if not news:
            news = _search_news(ticker)

        if not news:
            return f"No news found for {ticker}"

        # Parse date range for filtering
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        news_str = ""
        filtered_count = 0

        for article in news:
            data = _extract_article_data(article)

            # Filter by date if publish time is available
            if data["pub_date"]:
                pub_date_naive = data["pub_date"].replace(tzinfo=None)
                if not (start_dt <= pub_date_naive <= end_dt + relativedelta(days=1)):
                    continue

            news_str += f"### {data['title']} (source: {data['publisher']})\n"
            if data["summary"]:
                news_str += f"{data['summary']}\n"
            if data["link"]:
                news_str += f"Link: {data['link']}\n"
            news_str += "\n"
            filtered_count += 1

        if filtered_count == 0:
            return f"No news found for {ticker} between {start_date} and {end_date}"

        return f"## {ticker} News, from {start_date} to {end_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching news for {ticker}: {str(e)}"


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
        limit: Maximum number of articles to return

    Returns:
        Formatted string containing global news articles
    """
    # Search queries for macro/global news
    search_queries = [
        "stock market economy",
        "Federal Reserve interest rates",
        "inflation economic outlook",
        "global markets trading",
    ]

    all_news = []
    seen_titles = set()

    try:
        for query in search_queries:
            search = yf_retry(lambda q=query: yf.Search(
                query=q,
                news_count=limit,
                enable_fuzzy_query=True,
            ))

            if search.news:
                for article in search.news:
                    # Handle both flat and nested structures
                    if "content" in article:
                        data = _extract_article_data(article)
                        title = data["title"]
                    else:
                        title = article.get("title", "")

                    # Deduplicate by title
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_news.append(article)

            if len(all_news) >= limit:
                break

        if not all_news:
            return f"No global news found for {curr_date}"

        # Calculate date range
        curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        start_dt = curr_dt - relativedelta(days=look_back_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        news_str = ""
        for article in all_news[:limit]:
            # Handle both flat and nested structures
            if "content" in article:
                data = _extract_article_data(article)
                # Skip articles published after curr_date (look-ahead guard)
                if data.get("pub_date"):
                    pub_naive = data["pub_date"].replace(tzinfo=None) if hasattr(data["pub_date"], "replace") else data["pub_date"]
                    if pub_naive > curr_dt + relativedelta(days=1):
                        continue
                title = data["title"]
                publisher = data["publisher"]
                link = data["link"]
                summary = data["summary"]
            else:
                title = article.get("title", "No title")
                publisher = article.get("publisher", "Unknown")
                link = article.get("link", "")
                summary = ""

            news_str += f"### {title} (source: {publisher})\n"
            if summary:
                news_str += f"{summary}\n"
            if link:
                news_str += f"Link: {link}\n"
            news_str += "\n"

        return f"## Global Market News, from {start_date} to {curr_date}:\n\n{news_str}"

    except Exception as e:
        return f"Error fetching global news: {str(e)}"
