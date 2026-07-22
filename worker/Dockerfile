# Two targets:
#   slim (CI/tests) — queue loop + geometry post-processing, no ML deps (~300 MB)
#   full (default)  — adds CPU torch + geoai for real inference (~multi-GB)
# `docker compose build worker` builds `full`; CI builds `--target slim`.

FROM python:3.12-slim AS slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb'))['project']; \
    print('\n'.join(d['dependencies'] + d['optional-dependencies']['dev']))" > /tmp/req.txt \
    && pip install -r /tmp/req.txt

COPY worker ./worker
COPY scripts ./scripts
COPY tests ./tests

CMD ["python", "-m", "worker.main"]


FROM slim AS full

# opencv-python-headless (a geoai dep) needs glib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch FIRST so the geoai install below sees torch satisfied and
# doesn't pull CUDA wheels. Swap this line for the default index on a GPU host.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb'))['project']; \
    print('\n'.join(d['optional-dependencies']['ml']))" > /tmp/req-ml.txt \
    && pip install -r /tmp/req-ml.txt

CMD ["python", "-m", "worker.main"]
