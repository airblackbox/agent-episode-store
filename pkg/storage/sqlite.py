"""
SQLite storage backend for episodes.

EL-2: SQLite storage backend with WAL mode

Uses aiosqlite for async access. WAL (Write-Ahead Logging) mode allows
concurrent reads while a write is happening — important for a service
that ingests episodes while dashboards query them.

The schema stores episodes as a row with JSON columns for steps and
metadata. This keeps the storage simple (one table) while still
letting us query by agent_id, status, and date range.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from pkg.models import Episode, EpisodeCreate, EpisodeStep, EpisodeSummary

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id     TEXT PRIMARY KEY,
    agent_id       TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running',
    steps          TEXT NOT NULL DEFAULT '[]',
    tools_used     TEXT NOT NULL DEFAULT '[]',
    total_tokens   INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_duration_ms INTEGER NOT NULL DEFAULT 0,
    step_count     INTEGER NOT NULL DEFAULT 0,
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_id);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);",
    "CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at);",
]

_INSERT = """
INSERT INTO episodes (
    episode_id, agent_id, status, steps, tools_used,
    total_tokens, total_cost_usd, total_duration_ms, step_count,
    started_at, ended_at, metadata
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

_SELECT_BY_ID = "SELECT * FROM episodes WHERE episode_id = ?;"

_SELECT_LIST = """
SELECT episode_id, agent_id, status, tools_used,
       total_tokens, total_cost_usd, total_duration_ms, step_count,
       started_at, ended_at
FROM episodes
WHERE 1=1
"""

_COUNT = "SELECT COUNT(*) FROM episodes WHERE 1=1"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class EpisodeStore:
    """Async SQLite-backed episode storage.

    Usage:
        store = EpisodeStore("episodes.db")
        await store.init()           # creates tables
        await store.save(episode)    # insert
        ep = await store.get("id")   # fetch one
        results = await store.list() # fetch many
        await store.close()
    """

    def __init__(self, db_path: str = "episodes.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Open database, enable WAL mode, create tables."""
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row

        # WAL mode: allows concurrent reads during writes
        await self._db.execute("PRAGMA journal_mode=WAL;")
        # Sync mode NORMAL is safe with WAL and faster than FULL
        await self._db.execute("PRAGMA synchronous=NORMAL;")

        await self._db.execute(_CREATE_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await self._db.execute(idx_sql)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def save(self, episode: Episode) -> Episode:
        """Insert an episode into the database.

        Computes aggregates from steps before saving.
        Returns the episode with computed fields.
        """
        assert self._db is not None, "Call init() first"

        episode.compute_aggregates()

        steps_json = json.dumps(
            [s.model_dump(mode="json") for s in episode.steps]
        )
        tools_json = json.dumps(episode.tools_used)
        meta_json = json.dumps(episode.metadata)
        started = episode.started_at.isoformat()
        ended = episode.ended_at.isoformat() if episode.ended_at else None

        await self._db.execute(
            _INSERT,
            (
                episode.episode_id,
                episode.agent_id,
                episode.status.value,
                steps_json,
                tools_json,
                episode.total_tokens,
                episode.total_cost_usd,
                episode.total_duration_ms,
                episode.step_count,
                started,
                ended,
                meta_json,
            ),
        )
        await self._db.commit()
        return episode

    async def create(self, payload: EpisodeCreate) -> Episode:
        """Create a new episode from an ingest payload.

        Assigns episode_id and timestamps, then saves.
        """
        now = datetime.now(timezone.utc)
        ended = now if payload.status != "running" else None

        episode = Episode(
            agent_id=payload.agent_id,
            status=payload.status,
            steps=payload.steps,
            started_at=now,
            ended_at=ended,
            metadata=payload.metadata,
        )
        return await self.save(episode)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, episode_id: str) -> Episode | None:
        """Fetch a single episode by ID. Returns None if not found."""
        assert self._db is not None, "Call init() first"

        cursor = await self._db.execute(_SELECT_BY_ID, (episode_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_episode(row)

    async def list(
        self,
        agent_id: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EpisodeSummary]:
        """List episodes with optional filters.

        Returns lightweight summaries (no steps) for performance.
        """
        assert self._db is not None, "Call init() first"

        where_clauses: list[str] = []
        params: list[str | int] = []

        if agent_id:
            where_clauses.append("AND agent_id = ?")
            params.append(agent_id)
        if status:
            where_clauses.append("AND status = ?")
            params.append(status)
        if since:
            where_clauses.append("AND started_at >= ?")
            params.append(since.isoformat())
        if until:
            where_clauses.append("AND started_at <= ?")
            params.append(until.isoformat())

        query = (
            _SELECT_LIST
            + " ".join(where_clauses)
            + " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [self._row_to_summary(r) for r in rows]

    async def count(
        self,
        agent_id: str | None = None,
        status: str | None = None,
    ) -> int:
        """Count episodes matching filters."""
        assert self._db is not None, "Call init() first"

        where_clauses: list[str] = []
        params: list[str] = []

        if agent_id:
            where_clauses.append("AND agent_id = ?")
            params.append(agent_id)
        if status:
            where_clauses.append("AND status = ?")
            params.append(status)

        query = _COUNT + " " + " ".join(where_clauses)
        cursor = await self._db.execute(query, params)
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_episode(row: aiosqlite.Row) -> Episode:
        """Convert a database row to an Episode model."""
        steps_data = json.loads(row["steps"])
        steps = [EpisodeStep(**s) for s in steps_data]

        return Episode(
            episode_id=row["episode_id"],
            agent_id=row["agent_id"],
            status=row["status"],
            steps=steps,
            tools_used=json.loads(row["tools_used"]),
            total_tokens=row["total_tokens"],
            total_cost_usd=row["total_cost_usd"],
            total_duration_ms=row["total_duration_ms"],
            step_count=row["step_count"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_summary(row: aiosqlite.Row) -> EpisodeSummary:
        """Convert a database row to a lightweight summary."""
        return EpisodeSummary(
            episode_id=row["episode_id"],
            agent_id=row["agent_id"],
            status=row["status"],
            tools_used=json.loads(row["tools_used"]),
            total_tokens=row["total_tokens"],
            total_cost_usd=row["total_cost_usd"],
            total_duration_ms=row["total_duration_ms"],
            step_count=row["step_count"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
        )