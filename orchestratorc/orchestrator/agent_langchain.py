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
from orchestrator.reasoning import create_reasoning_engine

logger = logging.getLogger("juriaid.agent")

# Create reasoning engine instance
reasoning_engine = create_reasoning_engine()


# ---------------------------------------------------------------------------
# Enhanced System Prompt with Reasoning Integration
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a **Senior Partner** at a top Sri Lankan law firm with advanced legal reasoning capabilities.

You work in collaboration with an advanced reasoning engine that provides:
- Rule-based case classification and analysis
- Strategic planning based on case urgency and complexity
- Quality control checks

Your role is to:
1. Consider the reasoning engine's analysis and recommendations
2. Apply your legal expertise to validate and enhance the plan
3. Execute the recommended tools
4. Synthesize results into comprehensive legal advice

Available tools:
- **search_law_statutes** – Search Sri Lankan statutes/Acts/Sections
- **get_statute_by_act** – Retrieve full Act text by ID
- **search_past_cases** – Find relevant court precedents
- **generate_legal_questions** – Generate client intake questions
- **summarize_case** – Produce structured case summary
- **update_knowledge_base** – Save case to knowledge base

Always cite specific statutes and case names in your analysis.
Be professional, precise, and grounded in Sri Lankan law.
"""


# ---------------------------------------------------------------------------
# State schema (unchanged)
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State shared across all nodes"""
    messages: Sequence[BaseMessage]
    case_text: str
    user_prompt: str
    reasoning_analysis: Dict[str, Any]  # NEW: Reasoning engine output
    next_action: str


# ---------------------------------------------------------------------------
# LLM setup (unchanged)
# ---------------------------------------------------------------------------

def _build_llm():
    """Build the LLM with tools bound"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=settings.GEMINI_API_KEY
    )
    return llm.bind_tools(ALL_TOOLS)


# ---------------------------------------------------------------------------
# Enhanced Graph Nodes
# ---------------------------------------------------------------------------

_tool_node = ToolNode(ALL_TOOLS)


async def reason_node(state: AgentState) -> dict:
    """
    Enhanced reasoning node that combines LLM reasoning with rule-based analysis
    """
    logger.info("[REASON NODE] Starting hybrid reasoning analysis...")
    
    messages = list(state["messages"])
    case_text = state.get("case_text", "")
    user_prompt = state.get("user_prompt", "")
    
    # Step 1: Get hybrid reasoning engine analysis
    reasoning_output = reasoning_engine.analyze_and_plan(
        case_text=case_text,
        user_prompt=user_prompt,
        llm_analysis=None  # Will be enhanced in next iteration
    )
    
    logger.info(f"[REASON NODE] Case classified as: {reasoning_output['case_context']['category']}")
    logger.info(f"[REASON NODE] Urgency: {reasoning_output['case_context']['urgency']}")
    logger.info(f"[REASON NODE] Complexity: {reasoning_output['case_context']['complexity']:.2f}")
    logger.info(f"[REASON NODE] Recommended tools: {len(reasoning_output['recommended_tools'])}")
    
    # Step 2: Prepare enhanced context for LLM
    reasoning_context = f"""
### Reasoning Engine Analysis:

**Case Classification:**
- Category: {reasoning_output['case_context']['category']}
- Urgency: {reasoning_output['case_context']['urgency']}
- Complexity Score: {reasoning_output['case_context']['complexity']:.2f}

**Key Facts Identified:**
{chr(10).join(f'- {fact}' for fact in reasoning_output['case_context']['key_facts'][:5])}

**Legal Issues:**
{chr(10).join(f'- {issue}' for issue in reasoning_output['case_context']['legal_issues'])}

**Recommended Tool Sequence:**
{chr(10).join(f"{i+1}. {tool['tool']}: {tool['rationale']}" 
             for i, tool in enumerate(reasoning_output['recommended_tools']))}

**Strategic Reasoning:**
{chr(10).join(f'- {r}' for r in reasoning_output['reasoning'])}

Please validate this analysis and proceed with the recommended tools, or suggest modifications based on your legal expertise.
"""
    
    # Step 3: Add reasoning context to messages
    messages.insert(-1, SystemMessage(content=reasoning_context))
    
    # Step 4: Get LLM decision with enhanced context
    llm = _build_llm()
    response = await llm.ainvoke(messages)
    
    # Step 5: Store reasoning analysis in state
    return {
        "messages": messages + [response],
        "reasoning_analysis": reasoning_output,
        "next_action": "execute" if response.tool_calls else "synthesize"
    }


async def execute_node(state: AgentState) -> dict:
    """Execute tools (unchanged but logs reasoning context)"""
    logger.info("[EXECUTE NODE] Executing tools based on reasoning analysis...")
    
    reasoning_analysis = state.get("reasoning_analysis", {})
    if reasoning_analysis:
        logger.info(f"[EXECUTE NODE] Case context: {reasoning_analysis['case_context']['category']}")
    
    result = await _tool_node.ainvoke(state)
    return result


async def synthesize_node(state: AgentState) -> dict:
    """
    Enhanced synthesis with quality checks from reasoning engine
    """
    logger.info("[SYNTHESIZE NODE] Generating final response with quality validation...")
    
    messages = list(state["messages"])
    reasoning_analysis = state.get("reasoning_analysis", {})
    
    # Add quality check context
    quality_context = ""
    if reasoning_analysis and "quality_checks" in reasoning_analysis:
        quality_context = "\n\n### Quality Checks:\n"
        for check in reasoning_analysis["quality_checks"]:
            status = "✓ PASS" if check["passed"] else "✗ FAIL"
            quality_context += f"{status}: {check['message']}\n"
    
    synthesis_prompt = f"""Based on all the tool results above, provide a comprehensive legal analysis with:

1. **Executive Summary** - Brief overview of the case and findings
2. **Applicable Law** - Specific statutes and sections (with citations)
3. **Relevant Precedents** - Key cases (with names and citations)
4. **Legal Analysis** - Detailed reasoning
5. **Recommended Actions** - Next steps with priorities
6. **Client Questions** - Important questions to clarify (if generated)

{quality_context}

Format your response in structured markdown.
Be thorough, professional, and cite specific legal references.
"""
    
    messages.append(HumanMessage(content=synthesis_prompt))
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        google_api_key=settings.GEMINI_API_KEY
    )
    
    final_response = await llm.ainvoke(messages)
    
    logger.info("[SYNTHESIZE NODE] Final analysis complete")
    
    return {
        "messages": messages + [final_response],
        "next_action": "end"
    }


# ---------------------------------------------------------------------------
# Routing (unchanged)
# ---------------------------------------------------------------------------

def _after_reason(state: AgentState) -> Literal["execute", "synthesize"]:
    """Route after reasoning: execute tools or synthesize if no tools needed"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        logger.info(f"[ROUTER] Routing to EXECUTE ({len(last_message.tool_calls)} tool calls)")
        return "execute"
    logger.info("[ROUTER] No tools to execute, routing to SYNTHESIZE")
    return "synthesize"


# ---------------------------------------------------------------------------
# Build graph (unchanged structure)
# ---------------------------------------------------------------------------

def build_graph():
    """Build the LangGraph workflow"""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("reason", reason_node)
    graph.add_node("execute", execute_node)
    graph.add_node("synthesize", synthesize_node)
    
    # Set entry point
    graph.set_entry_point("reason")
    
    # Add edges
    graph.add_conditional_edges(
        "reason",
        _after_reason,
        {"execute": "execute", "synthesize": "synthesize"}
    )
    graph.add_edge("execute", "reason")  # Loop back for multi-step reasoning
    graph.add_edge("synthesize", END)
    
    return graph.compile()


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