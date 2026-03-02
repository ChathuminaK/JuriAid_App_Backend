import json
import logging
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory fallback store: session_id -> list of messages
_memory_store: dict[str, list[dict]] = {}

# Redis client (lazy init)
_redis = None
_redis_checked = False


def _get_redis():
    """Try to connect to Redis once. Returns client or None."""
    global _redis, _redis_checked

    if _redis_checked:
        return _redis

    _redis_checked = True

    if not settings.REDIS_ENABLED:
        logger.info("Redis disabled - using in-memory conversation store")
        return None

    try:
        import redis as redis_lib
        client = redis_lib.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        logger.info("Redis connected for long-term memory")
        _redis = client
        return _redis
    except Exception as e:
        logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
        return None


def save_conversation(session_id: str, role: str, content: str) -> None:
    """Save a message to conversation history (Redis or in-memory)."""
    message = {"role": role, "content": content[:2000]}
    redis_client = _get_redis()

    if redis_client:
        try:
            key = f"juriaid:memory:{session_id}"
            redis_client.rpush(key, json.dumps(message))
            redis_client.expire(key, 86400 * 7)  # 7 day TTL
            return
        except Exception as e:
            logger.warning(f"Redis save failed, using fallback: {e}")

    # In-memory fallback
    if session_id not in _memory_store:
        _memory_store[session_id] = []
    _memory_store[session_id].append(message)

    # Keep max 50 messages per session
    if len(_memory_store[session_id]) > 50:
        _memory_store[session_id] = _memory_store[session_id][-50:]


def get_conversation_history(session_id: str, last_n: int = 10) -> list[dict]:
    """Retrieve recent conversation history."""
    redis_client = _get_redis()

    if redis_client:
        try:
            key = f"juriaid:memory:{session_id}"
            messages = redis_client.lrange(key, -last_n, -1)
            return [json.loads(m) for m in messages]
        except Exception as e:
            logger.warning(f"Redis read failed, using fallback: {e}")

    # In-memory fallback
    history = _memory_store.get(session_id, [])
    return history[-last_n:]


def clear_conversation(session_id: str) -> None:
    """Clear conversation history for a session."""
    redis_client = _get_redis()

    if redis_client:
        try:
            redis_client.delete(f"juriaid:memory:{session_id}")
        except Exception:
            pass

    _memory_store.pop(session_id, None)