from typing import Optional

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
    get_global_news,
    get_disclosures,
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


def build_instrument_context(ticker: str, user_notes: str = "", user_position: str = "") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    ) + (user_position or "") + build_user_notes(user_notes)


def normalize_position(position: Optional[dict]) -> dict:
    """{"held": bool, "avg_price": float | None}; anything missing means "not held"."""
    position = position or {}
    held = bool(position.get("held"))
    avg = position.get("avg_price")
    try:
        avg = float(avg) if held and avg not in (None, "") and float(avg) > 0 else None
    except (TypeError, ValueError):
        avg = None
    return {"held": held, "avg_price": avg}


def format_price(value: float) -> str:
    """210000.0 -> '210,000'; 183.5 -> '183.5'; 0.12345 -> '0.1235'."""
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text or "0"


def build_position_context(position: Optional[dict]) -> str:
    """Prompt section stating whether the requester already owns the instrument.

    Without this, the rating scale ("reduce exposure", "maintain current position")
    silently assumes an existing holding. When no position is given the requester is
    treated as NOT holding it, so the advice is about whether to open a position.
    """
    pos = normalize_position(position)
    past_note = (
        " Any past decisions shown to you are earlier recommendations, not trades: do not "
        "assume they were executed. Base position-dependent advice only on the position "
        "stated here."
    )
    if not pos["held"]:
        return (
            "\n\n## Requester's current position\n"
            "The person requesting this analysis does NOT currently hold this instrument "
            "(no position). Frame every recommendation for someone deciding whether to open "
            "a new position: Buy = open a position now; Overweight = start a partial or "
            "staged entry; Hold = do not buy yet, keep watching; Underweight or Sell = do not "
            "buy / avoid. Do not recommend trimming, taking profits, or exiting a holding "
            "they do not have; give entry conditions (price levels, triggers) instead."
            + past_note + "\n"
        )
    cost = (
        f" Their average cost is {format_price(pos['avg_price'])} per share (user-provided; same "
        "currency as the quote). Mention the unrealized gain or loss against the current "
        "price, but do not let the purchase price (a sunk cost) drive the decision."
        if pos["avg_price"] else " Their average cost was not provided."
    )
    return (
        "\n\n## Requester's current position\n"
        "The person requesting this analysis currently HOLDS this instrument." + cost
        + " Frame every recommendation for an existing holder: Buy = add substantially; "
        "Overweight = add gradually; Hold = keep the current holding unchanged; "
        "Underweight = sell part of the holding; Sell = exit the holding."
        + past_note + "\n"
    )


def fence_untrusted(text: str) -> str:
    """Neutralise the <<< >>> fence markers inside untrusted text.

    Notes and channel posts are wrapped in <<< >>> so the model can tell data from
    instructions; text containing those markers could otherwise close the fence early
    and smuggle in instructions that look like they come from outside it.
    """
    return (text or "").replace("<<<", "‹‹‹").replace(">>>", "›››")


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
        "they state.\n<<<\n" + fence_untrusted(notes) + "\n>>>\n"
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


        
