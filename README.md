# agent-episode-store

Replayable episode ledger for AI agent runs. Part of [black-box-labworks](https://github.com/black-box-labworks).

Every agent task becomes an **episode** — a complete, replayable record of every LLM call, tool invocation, and decision the agent made. Episodes are the dataset that makes evals, policy enforcement, and reproducible debugging possible.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run
python -m cmd.server

# Or with Docker
docker build -t episode-store .
docker run -p 8100:8100 episode-store
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/episodes` | Ingest a new episode |
| GET | `/v1/episodes` | List episodes (with filters) |
| GET | `/v1/episodes/{id}` | Get full episode with steps |
| GET | `/v1/health` | Health check |

### Ingest an Episode

```bash
curl -X POST http://localhost:8100/v1/episodes \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "my-agent",
    "status": "success",
    "steps": [
      {"step_index": 0, "step_type": "llm_call", "model": "gpt-4", "tokens": 150},
      {"step_index": 1, "step_type": "tool_call", "tool_name": "web_search", "tokens": 200}
    ]
  }'
```

### Query Episodes

```bash
# All episodes
curl http://localhost:8100/v1/episodes

# Filter by agent
curl http://localhost:8100/v1/episodes?agent_id=my-agent

# Filter by status
curl http://localhost:8100/v1/episodes?status=failure
```

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `EPISODE_DB_PATH` | `episodes.db` | Path to SQLite database |
| `EPISODE_HOST` | `0.0.0.0` | Listen host |
| `EPISODE_PORT` | `8100` | Listen port |

## Testing

```bash
pip install -r requirements.txt
pytest -v
```

## License

Apache-2.0