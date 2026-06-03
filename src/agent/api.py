from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.status import HTTP_404_NOT_FOUND, HTTP_502_BAD_GATEWAY

from agent.executor import Error, safe_execute
from agent.loop import solve_sql

SERVED_DBS = {"concert_singer"}
app = FastAPI()


class QueryRequest(BaseModel):
    question: str
    db: str = "concert_singer"


class QueryResponse(BaseModel):
    sql: str
    rows: list[dict] | None = None
    error: str | None = None


def _db_path(db: str) -> str:
    if db not in SERVED_DBS:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"unknown database: {db}"
        )
    return f"sqlite-data/{db}/{db}.sqlite"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(req: QueryRequest) -> QueryResponse:
    db_path = _db_path(req.db)
    sql = solve_sql(req.question, db_path)
    if sql is None:
        raise HTTPException(
            status_code=HTTP_502_BAD_GATEWAY, detail="could not generate SQL"
        )
    result = safe_execute(sql, db_path)
    if isinstance(result, Error):
        return QueryResponse(sql=sql, error=result.message)
    return QueryResponse(sql=sql, rows=result.rows)
