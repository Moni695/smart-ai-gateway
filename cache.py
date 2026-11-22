"""
Response Cache module for LLMOps.
Provides an in-memory TTL cache with SHA-256 key hashing for LLM completions.
Returns cached responses instantly (0ms latency, $0 cost).
"""

import time
import json
import hashlib
from typing import Dict, Any, Optional, Tuple
from config import settings


class TTLCache:
    def __init__(self, default_ttl: int = None):
        self.default_ttl = default_ttl if default_ttl is not None else settings.CACHE_TTL_SECONDS
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

    def _generate_key(self, messages: list, model: str, temperature: float = 0.7, tools: list = None) -> str:
        """Creates a stable SHA-256 hash key from messages, model, and parameters."""
        serialized = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": round(temperature, 2),
            "tools": tools or []
        }, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, messages: list, model: str, temperature: float = 0.7, tools: list = None) -> Optional[Dict[str, Any]]:
        """Retrieves cached response if valid and not expired."""
        if not settings.CACHE_ENABLED:
            return None
            
        key = self._generate_key(messages, model, temperature, tools)
        entry = self._cache.get(key)
        if not entry:
            return None

        expires_at, response = entry
        if time.time() > expires_at:
            # Expired
            del self._cache[key]
            return None

        # Return a copy with cache_hit flagged
        cached_copy = dict(response)
        cached_copy["cache_hit"] = True
        return cached_copy

    def set(self, messages: list, model: str, response: Dict[str, Any], temperature: float = 0.7, tools: list = None, ttl: int = None):
        """Caches response with specified or default TTL."""
        if not settings.CACHE_ENABLED:
            return

        key = self._generate_key(messages, model, temperature, tools)
        ttl_seconds = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl_seconds
        self._cache[key] = (expires_at, response)

    def clear(self):
        """Clears all cached entries."""
        self._cache.clear()

    def size(self) -> int:
        """Returns the number of non-expired cached entries."""
        now = time.time()
        # Clean expired keys on size check
        expired_keys = [k for k, (exp, _) in self._cache.items() if exp <= now]
        for k in expired_keys:
            del self._cache[k]
        return len(self._cache)


response_cache = TTLCache()
