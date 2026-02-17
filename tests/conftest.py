"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from pkg.api.routes import set_store
from pkg.storage import EpisodeStore


@pytest_asyncio.fixture
async def store():
    """Create a temporary SQLite store for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    s = EpisodeStore(path)
    await s.init()
    yield s
    await s.close()
    os.unlink(path)


@pytest_asyncio.fixture
async def client(store):
    """Create a test HTTP client with the store injected."""
    from app.server import app

    set_store(store)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c