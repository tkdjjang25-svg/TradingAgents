from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str, user_notes: str = "") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    ) + build_user_notes(user_notes)


def build_user_notes(user_notes: str) -> str:
    """Prompt section for notes typed by the person who requested the analysis.

    The notes are unverified claims (upcoming events, context the data tools miss).
    Agents must weigh them, attribute them, and not treat them as tool data.
    Returns "" when there are no notes, so prompts are unchanged.
    """
    notes = (user_notes or "").strip()
    if not notes:
        return ""
    return (
        "\n\n## User-provided notes (unverified)\n"
        "The person who requested this analysis supplied the notes below. They are not "
        "tool data and may be wrong or incomplete. Consider every point in them, say "
        "explicitly how each one affects your view (or why it does not), and label any "
        "figure taken from them as \"user-provided\". Do not invent details beyond what "
        "they state.\n<<<\n" + notes + "\n>>>\n"
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
