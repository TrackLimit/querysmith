"""Build the schema section for the prompt from retrieved tables."""

from agent.schema import Table, read_schema_map


def build_schema_prompt(tables: list[Table], db_path: str) -> str:
    ddl = read_schema_map(db_path)
    blocks = []
    for table in tables:
        block = ddl[table.name]
        if table.sample_values:
            samples = [
                f"{col}: {', '.join(vals)}" for col, vals in table.sample_values.items()
            ]
            block += "\n-- sample values: " + "; ".join(samples)
        blocks.append(block)
    return "\n\n".join(blocks)
