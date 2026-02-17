"""
Episode API routes.

EL-3: Ingest endpoint — POST /v1/episodes from gateway webhook

Endpoints:
    POST /v1/episodes          — Ingest a new episode
    GET  /v1/episodes          — List episodes with filters
    GET  /v1/episodes/{id}     — Get a single episode with all steps
    GET  /v1/health            — Health check
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

from pkg.models import Episode, EpisodeCreate, EpisodeSummary
from pkg.storage import EpisodeStore

router = APIRouter()

# The store gets attached at startup (see cmd/server.py)
_store: EpisodeStore | None = None


def set_store(store: EpisodeStore) -> None:
    """Called at app startup to inject the store dependency."""
    global _store
    _store = store


def get_store() -> EpisodeStore:
    """Get the store, raising if not initialized."""
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/v1/health")
async def health() -> dict:
    """Health check — returns ok if the store is connected."""
    store = get_store()
    count = await store.count()
    return {
        "status": "ok",
        "service": "agent-episode-store",
        "version": "0.1.0",
        "episodes_stored": count,
    }


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@router.post("/v1/episodes", status_code=201, response_model=Episode)
async def create_episode(payload: EpisodeCreate) -> Episode:
    """Ingest a new episode.

    This is the main entry point for the gateway webhook.
    The gateway sends a complete episode (agent_id, steps, status)
    and the store assigns an episode_id and computes aggregates.

    Returns the full episode with all computed fields.
    """
    store = get_store()
    episode = await store.create(payload)
    return episode


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@router.get("/v1/episodes", response_model=list[EpisodeSummary])
async def list_episodes(
    agent_id: str | None = Query(default=None, description="Filter by agent"),
    status: str | None = Query(default=None, description="Filter by status"),
    since: datetime | None = Query(default=None, description="Episodes after this time"),
    until: datetime | None = Query(default=None, description="Episodes before this time"),
    limit: int = Query(default=50, ge=1, le=500, description="Max results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
) -> list[EpisodeSummary]:
    """List episodes with optional filters.

    Returns lightweight summaries (no steps) for performance.
    Use GET /v1/episodes/{id} for full episode with steps.
    """
    store = get_store()
    return await store.list(
        agent_id=agent_id,
        status=status,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )


@router.get("/v1/episodes/{episode_id}", response_model=Episode)
async def get_episode(episode_id: str) -> Episode:
    """Get a single episode by ID, including all steps.

    Returns 404 if the episode doesn't exist.
    """
    store = get_store()
    episode = await store.get(episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return episode