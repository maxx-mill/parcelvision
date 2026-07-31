# Build stages, ordered so a worker code change never invalidates the multi-GB
# torch/geoai layers (that slow rebuild is what tempted hot-copying code into a
# running container — which then ran with stale in-memory modules):
#   deps   — light deps only (cached across code edits)
#   mldeps — CPU torch + geoai on top of deps, INDEPENDENT of app code
#   slim   — deps + code (CI/tests; queue loop + geometry, no ML) (~300 MB)
#   full   — mldeps + code (default; real inference) (~multi-GB)
# `docker compose build worker` builds `full`; CI builds `--target slim`.

FROM python:3.12-slim AS deps

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml ./
RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb'))['project']; \
    print('\n'.join(d['dependencies'] + d['optional-dependencies']['dev']))" > /tmp/req.txt \
    && pip install -r /tmp/req.txt


FROM deps AS mldeps

# opencv-python-headless (a geoai dep) needs glib; rasterio/rioxarray's bundled
# GDAL needs system libexpat.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch FIRST so the geoai install below sees torch satisfied and
# doesn't pull CUDA wheels. Swap this line for the default index on a GPU host.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN python -c "import tomllib; d = tomllib.load(open('pyproject.toml','rb'))['project']; \
    print('\n'.join(d['optional-dependencies']['ml']))" > /tmp/req-ml.txt \
    && pip install -r /tmp/req-ml.txt


# Code is copied LAST in each runnable image, so editing it rebuilds only these
# cheap COPY layers — never the deps above.
FROM deps AS slim
COPY worker ./worker
COPY scripts ./scripts
COPY tests ./tests
CMD ["python", "-m", "worker.main"]


FROM mldeps AS full
COPY worker ./worker
COPY scripts ./scripts
COPY tests ./tests
CMD ["python", "-m", "worker.main"]
