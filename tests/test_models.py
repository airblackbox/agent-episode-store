"""
Tests for EL-1: Episode schema.

Validates that the Episode, EpisodeStep, and EpisodeCreate models
enforce types, defaults, and aggregate computation correctly.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pkg.models import Episode, EpisodeCreate, EpisodeStep, EpisodeStatus, StepType


# ---------------------------------------------------------------------------
# EpisodeStep
# ---------------------------------------------------------------------------

class TestEpisodeStep:
    def test_minimal_step(self):
        """A step only requires step_index and step_type."""
        step = EpisodeStep(step_index=0, step_type=StepType.LLM_CALL)
        assert step.step_index == 0
        assert step.step_type == StepType.LLM_CALL
        assert step.tokens == 0
        assert step.cost_usd == 0.0
        assert step.tool_name is None

    def test_full_step(self):
        """A step with all fields populated."""
        step = EpisodeStep(
            step_index=1,
            step_type=StepType.TOOL_CALL,
            air_record_id="abc-123",
            tool_name="web_search",
            model="gpt-4",
            provider="openai",
            input_summary="Search for weather",
            output_summary="72F and sunny",
            tokens=500,
            cost_usd=0.015,
            duration_ms=1200,
            metadata={"query": "weather today"},
        )
        assert step.tool_name == "web_search"
        assert step.tokens == 500

    def test_negative_tokens_rejected(self):
        """Tokens must be >= 0."""
        with pytest.raises(ValidationError):
            EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, tokens=-1)

    def test_negative_step_index_rejected(self):
        """Step index must be >= 0."""
        with pytest.raises(ValidationError):
            EpisodeStep(step_index=-1, step_type=StepType.LLM_CALL)


# ---------------------------------------------------------------------------
# EpisodeCreate
# ---------------------------------------------------------------------------

class TestEpisodeCreate:
    def test_minimal_create(self):
        """Only agent_id is required."""
        payload = EpisodeCreate(agent_id="test-agent")
        assert payload.agent_id == "test-agent"
        assert payload.status == EpisodeStatus.RUNNING
        assert payload.steps == []

    def test_empty_agent_id_rejected(self):
        """agent_id must not be empty."""
        with pytest.raises(ValidationError):
            EpisodeCreate(agent_id="")

    def test_create_with_steps(self):
        """Steps are accepted in the create payload."""
        payload = EpisodeCreate(
            agent_id="agent-1",
            steps=[
                EpisodeStep(step_index=0, step_type=StepType.LLM_CALL, tokens=100),
                EpisodeStep(step_index=1, step_type=StepType.TOOL_CALL, tool_name="calc"),
            ],
            status=EpisodeStatus.SUCCESS,
        )
        assert len(payload.steps) == 2
        assert payload.status == EpisodeStatus.SUCCESS


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_auto_generates_id(self):
        """episode_id is auto-generated if not provided."""
        ep = Episode(agent_id="agent-1")
        assert ep.episode_id is not None
        assert len(ep.episode_id) == 36  # UUID format

    def test_compute_aggregates(self):
        """compute_aggregates sums tokens, cost, duration and deduplicates tools."""
        ep = Episode(
            agent_id="agent-1",
            steps=[
                EpisodeStep(
                    step_index=0, step_type=StepType.LLM_CALL,
                    tokens=100, cost_usd=0.003, duration_ms=500,
                ),
                EpisodeStep(
                    step_index=1, step_type=StepType.TOOL_CALL,
                    tool_name="web_search",
                    tokens=200, cost_usd=0.006, duration_ms=800,
                ),
                EpisodeStep(
                    step_index=2, step_type=StepType.TOOL_CALL,
                    tool_name="web_search",  # duplicate tool
                    tokens=150, cost_usd=0.005, duration_ms=600,
                ),
                EpisodeStep(
                    step_index=3, step_type=StepType.TOOL_CALL,
                    tool_name="calculator",
                    tokens=50, cost_usd=0.001, duration_ms=100,
                ),
            ],
        )
        ep.compute_aggregates()

        assert ep.step_count == 4
        assert ep.total_tokens == 500
        assert ep.total_cost_usd == 0.015
        assert ep.total_duration_ms == 2000
        # web_search should only appear once
        assert ep.tools_used == ["web_search", "calculator"]

    def test_compute_aggregates_empty(self):
        """Aggregates for an episode with no steps."""
        ep = Episode(agent_id="agent-1")
        ep.compute_aggregates()
        assert ep.step_count == 0
        assert ep.total_tokens == 0
        assert ep.tools_used == []

    def test_status_enum_values(self):
        """All expected status values exist."""
        assert EpisodeStatus.RUNNING == "running"
        assert EpisodeStatus.SUCCESS == "success"
        assert EpisodeStatus.FAILURE == "failure"
        assert EpisodeStatus.TIMEOUT == "timeout"
        assert EpisodeStatus.KILLED == "killed"

    def test_step_type_enum_values(self):
        """All expected step types exist."""
        assert StepType.LLM_CALL == "llm_call"
        assert StepType.TOOL_CALL == "tool_call"
        assert StepType.TOOL_RESULT == "tool_result"
        assert StepType.DECISION == "decision"
        assert StepType.ERROR == "error"