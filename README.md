# Querysmith

Natural-language questions over a SQLite database: ask in English, get an answer
backed by a generated, executed SQL query.

```bash
uv sync
uv run python -m agent.cli "How many singers are there?"
```