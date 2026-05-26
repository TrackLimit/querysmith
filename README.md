# Querysmith

Natural-language questions over a SQLite database. The schema and its categorical
values are embedded into a Chroma index; each question retrieves the relevant tables,
which are formatted (DDL + sample values) into the prompt, and the model writes SQL
that runs against a read-only connection.

```bash
uv sync
export ANTHROPIC_API_KEY=sk-ant-...                # the CLI calls the Anthropic API

uv run python -m agent.ingest <db_path>            # build the index (once per DB)
uv run python -m agent.cli "<question>" <db_path>
```
