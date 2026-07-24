FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install deps from pyproject in their own layer so code edits don't bust the cache.
COPY pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb'))['project']; \
    print('\n'.join(d['dependencies'] + d['optional-dependencies']['dev']))" > /tmp/req.txt \
    && pip install -r /tmp/req.txt

COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
COPY scripts ./scripts
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
