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

## Evaluation

Execution accuracy on a 61-case Spider subset (52 Spider dev + 9 hand-written
adversarial), scored by result-set match against the gold query.

| difficulty | accuracy |
|---|---|
| easy | 81% (17/21) |
| medium | 69% (11/16) |
| hard | 15% (2/13) |
| extra | 55% (6/11) |
| **all** | **59% (36/61)** |

valid-SQL rate: 100% (every generated query ran without error). Generation is
non-deterministic — `temperature` isn't settable on this model — so the score
wobbles a couple of points run to run. This is the current baseline; later
improvements are measured against it.
