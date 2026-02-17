"""
Tests for EL-3: Ingest endpoint — POST /v1/episodes.

Also tests GET /v1/episodes, GET /v1/episodes/{id}, and GET /v1/health.
Uses httpx AsyncClient with FastAPI's TestClient pattern.
"""

from __future__ import annotations

import pytest


def _make_episode_payload(
    agent_id: str = "test-agent",
    status: str = "success",
    steps: list | None = None,
    metadata: dict | None = None,
) -> dict:
    """Helper to build a valid episode payload."""
    if steps is None:
        steps = [
            {
                "step_index": 0,
                "step_type": "llm_call",
                "model": "gpt-4",
                "provider": "openai",
                "tokens": 150,
                "cost_usd": 0.005,
                "duration_ms": 800,
            },
            {
                "step_index": 1,
                "step_type": "tool_call",
                "tool_name": "web_search",
                "tokens": 200,
                "cost_usd": 0.006,
                "duration_ms": 1200,
            },
        ]
    return {
        "agent_id": agent_id,
        "status": status,
        "steps": steps,
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Health endpoint returns ok."""
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "agent-episode-store"


# ---------------------------------------------------------------------------
# POST /v1/episodes
# ---------------------------------------------------------------------------

class TestIngest:
    @pytest.mark.asyncio
    async def test_create_episode(self, client):
        """POST creates an episode and returns 201."""
        payload = _make_episode_payload()
        resp = await client.post("/v1/episodes", json=payload)
        assert resp.status_code == 201

        data = resp.json()
        assert data["agent_id"] == "test-agent"
        assert data["status"] == "success"
        assert data["step_count"] == 2
        assert data["total_tokens"] == 350
        assert data["tools_used"] == ["web_search"]
        assert data["episode_id"] is not None

    @pytest.mark.asyncio
    async def test_create_minimal_episode(self, client):
        """POST with only agent_id works."""
        resp = await client.post("/v1/episodes", json={"agent_id": "minimal-agent"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_id"] == "minimal-agent"
        assert data["status"] == "running"
        assert data["step_count"] == 0

    @pytest.mark.asyncio
    async def test_create_rejects_empty_agent(self, client):
        """POST with empty agent_id returns 422."""
        resp = await client.post("/v1/episodes", json={"agent_id": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_with_metadata(self, client):
        """POST preserves metadata."""
        payload = _make_episode_payload(metadata={"experiment": "v2", "model_version": "4o"})
        resp = await client.post("/v1/episodes", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["metadata"]["experiment"] == "v2"

    @pytest.mark.asyncio
    async def test_create_computes_aggregates(self, client):
        """POST computes cost and duration from steps."""
        payload = _make_episode_payload()
        resp = await client.post("/v1/episodes", json=payload)
        data = resp.json()
        assert data["total_cost_usd"] == pytest.approx(0.011)
        assert data["total_duration_ms"] == 2000


# ---------------------------------------------------------------------------
# GET /v1/episodes/{id}
# ---------------------------------------------------------------------------

class TestGetEpisode:
    @pytest.mark.asyncio
    async def test_get_by_id(self, client):
        """GET returns the full episode with steps."""
        # Create first
        payload = _make_episode_payload()
        create_resp = await client.post("/v1/episodes", json=payload)
        ep_id = create_resp.json()["episode_id"]

        # Fetch
        resp = await client.get(f"/v1/episodes/{ep_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["episode_id"] == ep_id
        assert len(data["steps"]) == 2
        assert data["steps"][0]["step_type"] == "llm_call"
        assert data["steps"][1]["tool_name"] == "web_search"

    @pytest.mark.asyncio
    async def test_get_not_found(self, client):
        """GET returns 404 for nonexistent episode."""
        resp = await client.get("/v1/episodes/does-not-exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/episodes
# ---------------------------------------------------------------------------

class TestListEpisodes:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        """GET returns empty list when no episodes exist."""
        resp = await client.get("/v1/episodes")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client):
        """GET returns episodes after creating them."""
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="a1"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="a2"))

        resp = await client.get("/v1/episodes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # Summaries should NOT include steps
        assert "steps" not in data[0]

    @pytest.mark.asyncio
    async def test_list_filter_by_agent(self, client):
        """GET filters by agent_id query param."""
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="alpha"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="beta"))
        await client.post("/v1/episodes", json=_make_episode_payload(agent_id="alpha"))

        resp = await client.get("/v1/episodes", params={"agent_id": "alpha"})
        data = resp.json()
        assert len(data) == 2
        assert all(d["agent_id"] == "alpha" for d in data)

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, client):
        """GET filters by status query param."""
        await client.post("/v1/episodes", json=_make_episode_payload(status="success"))
        await client.post("/v1/episodes", json=_make_episode_payload(status="failure"))

        resp = await client.get("/v1/episodes", params={"status": "failure"})
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "failure"

    @pytest.mark.asyncio
    async def test_list_pagination(self, client):
        """GET supports limit and offset."""
        for i in range(5):
            await client.post("/v1/episodes", json=_make_episode_payload(agent_id=f"a{i}"))

        resp = await client.get("/v1/episodes", params={"limit": 2, "offset": 0})
        assert len(resp.json()) == 2

        resp2 = await client.get("/v1/episodes", params={"limit": 2, "offset": 2})
        assert len(resp2.json()) == 2