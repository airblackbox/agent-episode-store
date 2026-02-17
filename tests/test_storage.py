"""
Tests for EL-2: SQLite storage backend with WAL mode.

Validates save, get, list, count, and WAL mode activation.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pkg.models import Episode, EpisodeCreate, EpisodeStep, EpisodeStatus, StepType


# ---------------------------------------------------------------------------
# WAL mode
# ---------------------------------------------------------------------------

class TestWALMode:
    @pytest.mark.asyncio
    async def test_wal_enabled(self, store):
        """Database should be in WAL journal mode."""
        cursor = await store._db.execute("PRAGMA journal_mode;")
        row = await cursor.fetchone()
        assert row[0] == "wal"


# ---------------------------------------------------------------------------
# Save & Get
# ---------------------------------------------------------------------------

class TestSaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_retrieve(self, store):
        """Save an episode and get it back by ID."""
        ep = Episode(
            agent_id="agent-1",
            status=EpisodeStatus.SUCCESS,
            steps=[
                EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, tokens=100),
            ],
        )
        saved = await store.save(ep)
        assert saved.step_count == 1
        assert saved.total_tokens == 100

        fetched = await store.get(saved.episode_id)
        assert fetched is not None
        assert fetched.episode_id == saved.episode_id
        assert fetched.agent_id == "agent-1"
        assert len(fetched.steps) == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Getting a nonexistent episode returns None."""
        result = await store.get("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_from_payload(self, store):
        """Create an episode from an EpisodeCreate payload."""
        payload = EpisodeCreate(
            agent_id="agent-2",
            steps=[
                EpisodeStep(
                    step_index=0, step_type=StepType.TOOL_CALL,
                    tool_name="web_search", tokens=200, cost_usd=0.006,
                ),
            ],
            status=EpisodeStatus.SUCCESS,
            metadata={"experiment": "v1"},
        )
        ep = await store.create(payload)
        assert ep.episode_id is not None
        assert ep.agent_id == "agent-2"
        assert ep.total_tokens == 200
        assert ep.tools_used == ["web_search"]
        assert ep.metadata == {"experiment": "v1"}

    @pytest.mark.asyncio
    async def test_steps_roundtrip(self, store):
        """Steps survive JSON serialization/deserialization."""
        ep = Episode(
            agent_id="agent-3",
            steps=[
                EpisodeStep(
                    step_index=0, step_type=StepType.LLM_CALL,
                    model="gpt-4", provider="openai",
                    input_summary="Hello", output_summary="Hi there",
                    tokens=50, cost_usd=0.001, duration_ms=300,
                    metadata={"temperature": 0.7},
                ),
                EpisodeStep(
                    step_index=1, step_type=StepType.TOOL_CALL,
                    tool_name="calculator",
                    tokens=10, duration_ms=50,
                ),
            ],
        )
        saved = await store.save(ep)
        fetched = await store.get(saved.episode_id)

        assert len(fetched.steps) == 2
        assert fetched.steps[0].model == "gpt-4"
        assert fetched.steps[0].metadata == {"temperature": 0.7}
        assert fetched.steps[1].tool_name == "calculator"


# ---------------------------------------------------------------------------
# List & Count
# ---------------------------------------------------------------------------

class TestListAndCount:
    @pytest.mark.asyncio
    async def test_list_empty(self, store):
        """List returns empty when no episodes exist."""
        results = await store.list()
        assert results == []

    @pytest.mark.asyncio
    async def test_list_with_agent_filter(self, store):
        """List filters by agent_id."""
        await store.create(EpisodeCreate(agent_id="agent-a", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-b", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-a", status=EpisodeStatus.FAILURE))

        results = await store.list(agent_id="agent-a")
        assert len(results) == 2
        assert all(r.agent_id == "agent-a" for r in results)

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, store):
        """List filters by status."""
        await store.create(EpisodeCreate(agent_id="agent-1", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-1", status=EpisodeStatus.FAILURE))

        results = await store.list(status="success")
        assert len(results) == 1
        assert results[0].status == EpisodeStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_list_pagination(self, store):
        """Limit and offset work correctly."""
        for i in range(5):
            await store.create(EpisodeCreate(agent_id=f"agent-{i}", status=EpisodeStatus.SUCCESS))

        page1 = await store.list(limit=2, offset=0)
        page2 = await store.list(limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0].episode_id != page2[0].episode_id

    @pytest.mark.asyncio
    async def test_count(self, store):
        """Count returns correct totals."""
        assert await store.count() == 0

        await store.create(EpisodeCreate(agent_id="agent-1", status=EpisodeStatus.SUCCESS))
        await store.create(EpisodeCreate(agent_id="agent-2", status=EpisodeStatus.FAILURE))

        assert await store.count() == 2
        assert await store.count(agent_id="agent-1") == 1
        assert await store.count(status="failure") == 1