"""
Storage module for the AI Gateway.
Persists every incoming LLM request and metadata into SQLite using aiosqlite.
"""

import aiosqlite
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import settings


class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self._initialized = False

    async def init_db(self):
        """Initializes database schema if not already created."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS request_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    route_policy TEXT NOT NULL,
                    model_requested TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    cache_hit INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    fallback_used INTEGER NOT NULL DEFAULT 0,
                    routing_reason TEXT,
                    complexity_score REAL DEFAULT 0.0
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_req_time ON request_logs(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_req_policy ON request_logs(route_policy)")
            await db.commit()
            self._initialized = True

    async def _ensure_init(self):
        if not self._initialized:
            await self.init_db()

    async def log_request(
        self,
        request_id: str,
        route_policy: str,
        model_requested: str,
        model_used: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        cost_usd: float,
        cache_hit: bool,
        status: str,
        error_message: Optional[str] = None,
        fallback_used: bool = False,
        routing_reason: Optional[str] = None,
        complexity_score: float = 0.0,
    ) -> int:
        """Inserts a single request log entry."""
        await self._ensure_init()
        total_tokens = prompt_tokens + completion_tokens
        ts = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO request_logs (
                    request_id, timestamp, route_policy, model_requested, model_used,
                    prompt_tokens, completion_tokens, total_tokens, latency_ms,
                    cost_usd, cache_hit, status, error_message, fallback_used,
                    routing_reason, complexity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id, ts, route_policy, model_requested, model_used,
                    prompt_tokens, completion_tokens, total_tokens, latency_ms,
                    cost_usd, int(cache_hit), status, error_message, int(fallback_used),
                    routing_reason, complexity_score
                )
            )
            await db.commit()
            return cursor.lastrowid

    async def get_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        policy: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves paginated request logs with optional filtering."""
        await self._ensure_init()
        query = "SELECT * FROM request_logs WHERE 1=1"
        params = []
        if policy:
            query += " AND route_policy = ?"
            params.append(policy)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_metrics(self) -> Dict[str, Any]:
        """Calculates aggregated LLMOps metrics across all logged requests."""
        await self._ensure_init()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Overall summary
            cursor = await db.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(total_tokens) as total_tokens,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(cost_usd) as total_cost_usd,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(cache_hit) as total_cache_hits,
                    SUM(fallback_used) as total_fallbacks,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as total_errors
                FROM request_logs
            """)
            overall = dict(await cursor.fetchone() or {})
            total_reqs = overall.get("total_requests") or 0
            cache_hits = overall.get("total_cache_hits") or 0
            fallbacks = overall.get("total_fallbacks") or 0
            errors = overall.get("total_errors") or 0

            # Policy breakdown
            cursor = await db.execute("""
                SELECT 
                    route_policy,
                    COUNT(*) as req_count,
                    SUM(total_tokens) as tokens,
                    SUM(cost_usd) as cost,
                    AVG(latency_ms) as avg_latency
                FROM request_logs
                GROUP BY route_policy
            """)
            policy_rows = [dict(r) for r in await cursor.fetchall()]

            # Model breakdown
            cursor = await db.execute("""
                SELECT 
                    model_used,
                    COUNT(*) as count,
                    SUM(cost_usd) as cost,
                    AVG(latency_ms) as avg_latency
                FROM request_logs
                GROUP BY model_used
            """)
            model_rows = [dict(r) for r in await cursor.fetchall()]

            return {
                "total_requests": total_reqs,
                "total_tokens": overall.get("total_tokens") or 0,
                "total_prompt_tokens": overall.get("total_prompt_tokens") or 0,
                "total_completion_tokens": overall.get("total_completion_tokens") or 0,
                "total_cost_usd": round(overall.get("total_cost_usd") or 0.0, 6),
                "avg_latency_ms": round(overall.get("avg_latency_ms") or 0.0, 2),
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hits / total_reqs, 4) if total_reqs > 0 else 0.0,
                "fallbacks": fallbacks,
                "fallback_rate": round(fallbacks / total_reqs, 4) if total_reqs > 0 else 0.0,
                "errors": errors,
                "error_rate": round(errors / total_reqs, 4) if total_reqs > 0 else 0.0,
                "by_policy": policy_rows,
                "by_model": model_rows,
            }

    async def clear_logs(self):
        """Clears all logged requests (useful before benchmarks or in tests)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM request_logs")
            await db.commit()


db_manager = DatabaseManager()
