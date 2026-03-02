import logging
from typing import Optional
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory fallback store: session_id -> list of messages
_memory_store: dict[str, list[dict]] = {}


def _get_redis_client():
    """Try to connect to Redis. Returns client or None."""
    if not settings.REDIS_ENABLED:
        return None
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        logger.info("Redis connected for memory")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
        return None


_redis = _get_redis_client()


def save_conversation(session_id: str, role: str, content: str) -> None:
    """Save a message to conversation history."""
    import json

    message = {"role": role, "content": content[:2000]}  # Limit size

    if _redis:
        try:
            key = f"juriaid:memory:{session_id}"
            _redis.rpush(key, json.dumps(message))
            _redis.expire(key, 86400 * 7)  # 7 day TTL
            return
        except Exception as e:
            logger.warning(f"Redis save failed, using fallback: {e}")

    # In-memory fallback
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    _memory_store[session_id].append(message)

    # Keep max 50 messages per session in memory
    if len(_memory_store[session_id]) > 50:
        _memory_store[session_id] = _memory_store[session_id][-50:]


def get_conversation_history(session_id: str, last_n: int = 10) -> list[dict]:
    """Retrieve recent conversation history."""
    import json

    if _redis:
        try:
            key = f"juriaid:memory:{session_id}"
            messages = _redis.lrange(key, -last_n, -1)
            return [json.loads(m) for m in messages]
        except Exception as e:
            logger.warning(f"Redis read failed, using fallback: {e}")

    # In-memory fallback
    history = _memory_store.get(session_id, [])
    return history[-last_n:]


def clear_conversation(session_id: str) -> None:
    """Clear conversation history for a session."""
    if _redis:
        try:
            _redis.delete(f"juriaid:memory:{session_id}")
        except Exception:
            pass

    _memory_store.pop(session_id, None)