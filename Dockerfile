# syntax=docker/dockerfile:1

FROM dhi.io/python:3-debian13-dev AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /bin/uv
WORKDIR /app

# system python only, so the copied .venv resolves in the runtime stage
ENV UV_COMPILE_BYTECODE=1 UV_PYTHON_PREFERENCE=only-system UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY sqlite-data/concert_singer ./sqlite-data/concert_singer
RUN uv sync --frozen --no-editable

RUN /app/.venv/bin/python -c "from agent.retrieval import embed; embed(['warmup'])"
RUN /app/.venv/bin/python -m agent.ingest sqlite-data/concert_singer/concert_singer.sqlite

# stage onnxruntime's C++ libs, dereferencing the versioned symlinks
RUN mkdir /onnx-libs && cp -L /usr/lib/*/libstdc++.so.6 /usr/lib/*/libgcc_s.so.1 /onnx-libs/


FROM dhi.io/python:3
WORKDIR /app

COPY --from=build /onnx-libs/ /usr/lib/x86_64-linux-gnu/

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/sqlite-data /app/sqlite-data
# chroma writes a -wal/-shm beside the db, so the non-root user must own the index
COPY --from=build --chown=65532:65532 /app/chroma_db /app/chroma_db
COPY --from=build --chown=65532:65532 /tmp/fastembed_cache /tmp/fastembed_cache

EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
