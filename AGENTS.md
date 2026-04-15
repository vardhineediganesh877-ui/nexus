# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

NEXUS is a Python-based AI crypto trading intelligence platform. Single service: FastAPI REST API backed by SQLite (auto-created at `~/.nexus/data/trades.db`). No external databases or Docker required for development.

### Prerequisites

- Python 3.10+ (3.12 available on the VM)
- `~/.nexus/data` directory must exist before running tests (some tests and the execution engine create the SQLite DB there)

### Key commands

All standard commands are in the `Makefile`:

| Task | Command |
|------|---------|
| Install deps | `pip install -e ".[dev]" --break-system-packages` |
| Run tests | `python3 -m pytest tests/ -v` |
| Lint (ruff) | `ruff check src/` |
| Lint (black) | `black --check src/` |
| Dev server | `python3 -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload` |
| CLI | `python3 -m src.cli <command>` |

### Non-obvious gotchas

- `httpx` is needed for API tests (`FastAPI.testclient` requires it) but is not listed in `pyproject.toml` dev dependencies. Install it alongside dev deps: `pip install httpx --break-system-packages`.
- `~/.nexus/data` must be created before running tests — 8 telegram tests fail with `sqlite3.OperationalError` if this directory is missing. Run `mkdir -p ~/.nexus/data` before `pytest`.
- `/home/ubuntu/.local/bin` must be on `PATH` for `pytest`, `ruff`, `black`, `uvicorn`, and `nexus` CLI to be found. Add via `export PATH="/home/ubuntu/.local/bin:$PATH"`.
- Paper trading mode is the default (`NEXUS_PAPER=true`). No exchange API keys are needed for analysis, backtesting, or paper trades — only for live trade execution.
- The API fetches live exchange data via CCXT, so internet access is required for endpoints like `/api/v1/market/*/ticker` and `/api/v1/analyze/*`.
