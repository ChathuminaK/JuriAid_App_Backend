"""
Simple Redis cache for /api/analyze responses.
Key: juriaid:analysis:{user_id}:{file_hash}
TTL: configurable (default 24 hours)
"""
import hashlib
import json
import logging

logger = logging.getLogger("redis_cache")

_redis_client = None


def _get_client():
    global _redis_client
    if _redis_client is None:
        import redis
        from config import get_settings
        s = get_settings()
        _redis_client = redis.Redis(
            host=s.REDIS_HOST,
            port=s.REDIS_PORT,
            password=s.REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
        )
    return _redis_client


def _make_key(user_id, file_hash: str) -> str:
    return f"juriaid:analysis:{user_id}:{file_hash}"


def hash_file(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:24]


def get_cached(user_id, file_hash: str) -> dict | None:
    """Return cached analysis dict or None."""
    from config import get_settings
    if not get_settings().REDIS_ENABLED:
        return None
    try:
        raw = _get_client().get(_make_key(user_id, file_hash))
        if raw:
            logger.info(f"[RedisCache] Cache HIT | user={user_id}")
            return json.loads(raw)
    except Exception as e:
        logger.warning(f"[RedisCache] get failed (non-fatal): {e}")
    return None


def save_to_cache(user_id, file_hash: str, result: dict) -> None:
    """Save analysis dict to Redis with TTL."""
    from config import get_settings
    s = get_settings()
    if not s.REDIS_ENABLED:
        return
    try:
        _get_client().setex(
            _make_key(user_id, file_hash),
            s.REDIS_CACHE_TTL,
            json.dumps(result),
        )
        logger.info(f"[RedisCache] Saved | user={user_id} | TTL={s.REDIS_CACHE_TTL}s")
    except Exception as e:
        logger.warning(f"[RedisCache] save failed (non-fatal): {e}")


# ── Saved Reports (user-persistent, keyed by analysis_id) ──────────────────

def save_report(user_id, report: dict) -> bool:
    """Save an analysis report to user's saved reports (Redis Hash)."""
    from config import get_settings
    if not get_settings().REDIS_ENABLED:
        return False
    try:
        analysis_id = report.get("analysis_id", "")
        if not analysis_id:
            return False
        key = f"juriaid:saved_reports:{user_id}"
        _get_client().hset(key, analysis_id, json.dumps(report))
        logger.info(f"[RedisCache] Report saved | user={user_id} | id={analysis_id[:8]}")
        return True
    except Exception as e:
        logger.warning(f"[RedisCache] save_report failed (non-fatal): {e}")
        return False


def get_saved_reports(user_id) -> list:
    """Return all saved reports for a user, newest first."""
    from config import get_settings
    if not get_settings().REDIS_ENABLED:
        return []
    try:
        key = f"juriaid:saved_reports:{user_id}"
        all_fields = _get_client().hgetall(key)
        reports = []
        for _, json_str in all_fields.items():
            try:
                reports.append(json.loads(json_str))
            except Exception:
                pass
        reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return reports
    except Exception as e:
        logger.warning(f"[RedisCache] get_saved_reports failed (non-fatal): {e}")
        return []


def delete_saved_report(user_id, analysis_id: str) -> bool:
    """Delete a single saved report by analysis_id."""
    from config import get_settings
    if not get_settings().REDIS_ENABLED:
        return False
    try:
        key = f"juriaid:saved_reports:{user_id}"
        deleted = _get_client().hdel(key, analysis_id)
        if deleted:
            logger.info(f"[RedisCache] Report deleted | user={user_id} | id={analysis_id[:8]}")
        return bool(deleted)
    except Exception as e:
        logger.warning(f"[RedisCache] delete_saved_report failed (non-fatal): {e}")
        return False