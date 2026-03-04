"""
JuriAid Hybrid Memory Manager
------------------------------
Short-term memory: ConversationBufferWindow (last N messages, in-process)
Long-term memory : Redis (persistent across sessions, 7-day TTL)

Reference: Proposal Section - "Implement a memory system for ongoing conversations"
  - ConversationBufferWindowMemory for short-term context [6], [10]
  - Redis-backed ConversationBufferMemory for long-term case continuity [10], [11]
"""

import json
import logging
from collections import defaultdict
from config import get_settings

logger = logging.getLogger("memory_agent")
settings = get_settings()

# ============================================================
# SHORT-TERM MEMORY: ConversationBufferWindow (in-process)
# Keeps last N messages per session for immediate LLM context
# ============================================================

_short_term_store: dict[str, list[dict]] = defaultdict(list)


def _save_short_term(session_id: str, role: str, content: str) -> None:
    """Save message to short-term ConversationBuffer (windowed)."""
    _short_term_store[session_id].append({"role": role, "content": content})

    # Keep only last N messages (sliding window)
    window = settings.SHORT_TERM_WINDOW
    if len(_short_term_store[session_id]) > window:
        _short_term_store[session_id] = _short_term_store[session_id][-window:]


def _get_short_term(session_id: str) -> list[dict]:
    """Retrieve short-term ConversationBuffer messages."""
    return list(_short_term_store.get(session_id, []))


# ============================================================
# LONG-TERM MEMORY: Redis (persistent, survives restarts)
# Stores full conversation history for case continuity
# ============================================================

_redis_client = None
_redis_checked = False


def _get_redis():
    """Lazy-init Redis connection. Returns client or None."""
    global _redis_client, _redis_checked

    if _redis_checked:
        return _redis_client

    _redis_checked = True

    if not settings.REDIS_ENABLED:
        logger.info("[MemoryAgent] Long-term memory: Redis DISABLED — using in-memory fallback")
        return None

    try:
        import redis as redis_lib
        client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        logger.info("[MemoryAgent] Long-term memory: Redis CONNECTED ✅")
        _redis_client = client
        return _redis_client
    except ImportError:
        logger.warning("[MemoryAgent] Redis package not installed — pip install redis")
        return None
    except Exception as e:
        logger.warning(f"[MemoryAgent] Redis unavailable — using in-memory fallback: {e}")
        return None


# In-memory fallback for long-term when Redis is unavailable
_long_term_fallback: dict[str, list[dict]] = defaultdict(list)


def _save_long_term(session_id: str, role: str, content: str) -> None:
    """Save message to long-term memory (Redis or in-memory fallback)."""
    message = {"role": role, "content": content[:2000]}
    redis = _get_redis()

    if redis:
        try:
            key = f"juriaid:memory:{session_id}"
            redis.rpush(key, json.dumps(message))
            redis.expire(key, 86400 * settings.REDIS_TTL_DAYS)
            return
        except Exception as e:
            logger.warning(f"[MemoryAgent] Redis write failed, using fallback: {e}")

    # Fallback: in-memory long-term
    _long_term_fallback[session_id].append(message)
    max_msgs = settings.LONG_TERM_MAX_MESSAGES
    if len(_long_term_fallback[session_id]) > max_msgs:
        _long_term_fallback[session_id] = _long_term_fallback[session_id][-max_msgs:]


def _get_long_term(session_id: str, last_n: int = 20) -> list[dict]:
    """Retrieve long-term memory (Redis or in-memory fallback)."""
    redis = _get_redis()

    if redis:
        try:
            key = f"juriaid:memory:{session_id}"
            messages = redis.lrange(key, -last_n, -1)
            return [json.loads(m) for m in messages]
        except Exception as e:
            logger.warning(f"[MemoryAgent] Redis read failed, using fallback: {e}")

    # Fallback
    history = _long_term_fallback.get(session_id, [])
    return history[-last_n:]


# ============================================================
# PUBLIC API — Used by pipeline.py
# ============================================================

def save_conversation(session_id: str, role: str, content: str) -> None:
    """
    Save message to BOTH memory systems:
    - Short-term ConversationBuffer (for immediate LLM context window)
    - Long-term Redis/fallback (for case continuity across sessions)
    """
    if not session_id or not content:
        return

    _save_short_term(session_id, role, content)
    _save_long_term(session_id, role, content)

    logger.debug(
        f"[MemoryAgent] Saved {role} message | session={session_id[:8]}... | "
        f"short-term={len(_short_term_store.get(session_id, []))} msgs, "
        f"long-term=persisted"
    )


def get_conversation_history(session_id: str) -> str:
    """
    Get conversation context for LLM.
    Strategy: Use short-term buffer first (recent context).
    If empty, fall back to long-term memory (returning user).
    """
    if not session_id:
        return ""

    # Try short-term first (current session)
    short = _get_short_term(session_id)

    # If no short-term, check long-term (returning user with history)
    if not short:
        long = _get_long_term(session_id, last_n=10)
        if long:
            logger.info(
                f"[MemoryAgent] Restored {len(long)} messages from long-term memory "
                f"for session {session_id[:8]}..."
            )
            # Populate short-term from long-term for this session
            for msg in long:
                _save_short_term(session_id, msg["role"], msg["content"])
            short = _get_short_term(session_id)

    if not short:
        return ""

    # Format for LLM
    parts = []
    for msg in short:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")

    history = "\n".join(parts)
    logger.info(
        f"[MemoryAgent] Retrieved {len(short)} messages for LLM context | "
        f"session={session_id[:8]}..."
    )
    return history


def clear_conversation(session_id: str) -> None:
    """Clear both short-term and long-term memory for a session."""
    # Short-term
    _short_term_store.pop(session_id, None)

    # Long-term
    redis = _get_redis()
    if redis:
        try:
            redis.delete(f"juriaid:memory:{session_id}")
        except Exception:
            pass
    _long_term_fallback.pop(session_id, None)

    logger.info(f"[MemoryAgent] Cleared all memory for session {session_id[:8]}...")


def get_memory_status() -> dict:
    """Get memory system health status."""
    redis = _get_redis()
    redis_connected = False
    redis_info = "disabled"

    if settings.REDIS_ENABLED:
        if redis:
            try:
                redis.ping()
                redis_connected = True
                redis_info = "connected"
            except Exception:
                redis_info = "connection failed"
        else:
            redis_info = "connection failed"
    else:
        redis_info = "disabled (using in-memory fallback)"

    short_term_sessions = len(_short_term_store)
    long_term_sessions = len(_long_term_fallback) if not redis_connected else "stored in Redis"

    logger.info(
        f"[MemoryAgent] Health: Redis={redis_info}, "
        f"short-term sessions={short_term_sessions}"
    )

    return {
        "memory_system": "hybrid",
        "short_term": {
            "type": "ConversationBufferWindow",
            "framework": "LangChain-compatible",
            "window_size": settings.SHORT_TERM_WINDOW,
            "active_sessions": short_term_sessions,
            "status": "active",
        },
        "long_term": {
            "type": "Redis" if settings.REDIS_ENABLED else "in-memory fallback",
            "connected": redis_connected,
            "status": redis_info,
            "ttl_days": settings.REDIS_TTL_DAYS if settings.REDIS_ENABLED else "N/A",
            "active_sessions": long_term_sessions,
        },
    }