"""
JuriAid Orchestrator – LangGraph Agentic Workflow
===================================================
3-node graph:  reason  ─→  execute  ─→  reason  ─→  synthesize  ─→  END

* **reason** – the "Senior Partner" LLM decides which tools to call
  (or produces a final answer).
* **execute** – invokes the selected tools asynchronously.
* **synthesize** – formats the final structured response.

Chat history is persisted in Redis so multi-turn conversations just
work by passing the same ``session_id``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Literal, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from config import settings
from orchestrator.tools import ALL_TOOLS

logger = logging.getLogger("juriaid.agent")

# ---------------------------------------------------------------------------
# System prompt  (the "Senior Partner")
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a **Senior Partner** at a top Sri Lankan law firm.
You have access to the following specialised research tools:

1. **search_law_statutes** – Search Sri Lankan statutes / Acts / Sections.
2. **get_statute_by_act** – Get the full text of a specific Act by ID.
3. **search_past_cases** – Find relevant court precedents via hybrid semantic + citation search.
4. **generate_legal_questions** – Generate client-intake / case-preparation questions.
5. **summarize_case** – Produce a structured JSON summary of a case.
6. **update_knowledge_base** – Save a case to the local knowledge base.

### Instructions
- Think step-by-step about what information the user needs.
- Call **one or more tools** in parallel when appropriate.
- After receiving tool results, analyse them and produce a comprehensive
  **Executive Summary** that **cites specific statutes and case names**.
- If a tool returns an error or fallback data, mention that the specific
  data source was temporarily unavailable but still answer with whatever
  information is available.
- Always be professional, precise, and reference Sri Lankan law.
- When the user uploads a case document, start by summarising it, then
  search for relevant statutes and past cases, then generate questions.
"""

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    session_id: str
    case_text: str | None
    final_answer: str | None


# ---------------------------------------------------------------------------
# LLM + tool binding
# ---------------------------------------------------------------------------


def _build_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True,
    )
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

_tool_node = ToolNode(ALL_TOOLS)


async def reason_node(state: AgentState) -> dict:
    """The LLM reasons about the conversation and optionally calls tools."""
    llm = _build_llm()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


async def execute_node(state: AgentState) -> dict:
    """Run whatever tool calls the LLM requested."""
    result = await _tool_node.ainvoke(state)
    # ToolNode returns {"messages": [ToolMessage, ...]}
    return result


async def synthesize_node(state: AgentState) -> dict:
    """Extract the final AI message and store it as ``final_answer``."""
    last = state["messages"][-1]
    answer = last.content if isinstance(last, AIMessage) else str(last)
    return {"final_answer": answer}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _after_reason(state: AgentState) -> Literal["execute", "synthesize"]:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "execute"
    return "synthesize"


# ---------------------------------------------------------------------------
# Build the graph (compiled once, reused for every request)
# ---------------------------------------------------------------------------


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("reason", reason_node)
    g.add_node("execute", execute_node)
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("reason")
    g.add_conditional_edges("reason", _after_reason)
    g.add_edge("execute", "reason")
    g.add_edge("synthesize", END)

    return g.compile()


_graph = build_graph()


# ---------------------------------------------------------------------------
# Redis chat history helper
# ---------------------------------------------------------------------------


def get_session_history(session_id: str):
    """Return a ``RedisChatMessageHistory`` instance for *session_id*.

    Falls back to an in-memory list when Redis is not reachable.
    """
    try:
        from langchain_community.chat_message_histories import (
            RedisChatMessageHistory,
        )

        return RedisChatMessageHistory(session_id=session_id, url=settings.REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) – using in-memory history", exc)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_agent(
    query: str,
    session_id: str,
    case_text: str | None = None,
) -> Dict[str, Any]:
    """Run the LangGraph agent for a single turn.

    Parameters
    ----------
    query : str
        The user's natural-language question / instruction.
    session_id : str
        Unique conversation ID (used for Redis memory).
    case_text : str | None
        Optional raw document text (e.g. from a PDF upload).

    Returns
    -------
    dict  with keys: success, session_id, answer, tool_calls_made, timestamp
    """
    # ── load history ──
    history_store = get_session_history(session_id)
    past_messages: List[BaseMessage] = []
    if history_store is not None:
        try:
            past_messages = list(history_store.messages)
        except Exception:
            past_messages = []

    # ── build messages ──
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(past_messages)

    # If case_text supplied, prepend it as context
    if case_text:
        messages.append(
            HumanMessage(
                content=(
                    f"I am uploading a legal case document. Here is the extracted text:\n\n"
                    f"---BEGIN CASE---\n{case_text[:4000]}\n---END CASE---\n\n"
                    f"User instruction: {query}"
                )
            )
        )
    else:
        messages.append(HumanMessage(content=query))

    # ── invoke graph ──
    try:
        final_state = await _graph.ainvoke(
            {
                "messages": messages,
                "session_id": session_id,
                "case_text": case_text,
                "final_answer": None,
            }
        )

        answer = final_state.get("final_answer") or ""

        # ── collect which tools were called ──
        tool_calls_made: List[str] = []
        for msg in final_state["messages"]:
            if isinstance(msg, ToolMessage):
                tool_calls_made.append(msg.name)

        # ── persist to Redis ──
        if history_store is not None:
            try:
                user_msg = messages[-1]  # the HumanMessage we just added
                history_store.add_message(user_msg)
                history_store.add_message(AIMessage(content=answer))
            except Exception as exc:
                logger.warning("Could not persist to Redis: %s", exc)

        return {
            "success": True,
            "session_id": session_id,
            "answer": answer,
            "tool_calls_made": tool_calls_made,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as exc:
        logger.exception("Agent run failed")
        return {
            "success": False,
            "session_id": session_id,
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }


# ---------------------------------------------------------------------------
# Retrieve full chat history (for GET /api/chat/history)
# ---------------------------------------------------------------------------


def get_chat_history(session_id: str) -> List[Dict[str, str]]:
    store = get_session_history(session_id)
    if store is None:
        return []
    try:
        return [
            {"role": type(m).__name__, "content": m.content}
            for m in store.messages
        ]
    except Exception:
        return []