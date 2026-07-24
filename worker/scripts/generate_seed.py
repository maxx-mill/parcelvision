"""Generate data/seed/demo_stl.geojson by running the real pipeline once.

Run inside the worker-full container (needs geoai + network):

    docker compose run --rm --no-deps -v ./data/seed:/out worker \
        python scripts/generate_seed.py /out/demo_stl.geojson

The output is a FeatureCollection of genuinely detected footprints for the
demo AOI, with the backend recorded in properties — `make demo` then loads it
without any live inference.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker.backends import get_backend  # noqa: E402
from worker.pipeline.fetch import fetch_imagery  # noqa: E402
from worker.pipeline.postprocess import postprocess  # noqa: E402

from worker import config  # noqa: E402

# ~0.25 km² residential block in Clayton / University City, St. Louis County.
DEMO_BBOX = [-90.3167, 38.6465, -90.3111, 38.6501]


def main(out_path: str) -> None:
    backend = get_backend(config.inference_backend())
    with tempfile.TemporaryDirectory() as tmp:
        rasters = fetch_imagery(DEMO_BBOX, Path(tmp)) if backend.needs_imagery else []
        gdf = postprocess(backend.extract(rasters, DEMO_BBOX), DEMO_BBOX)

    fc = json.loads(gdf.to_json())
    fc["bbox"] = DEMO_BBOX
    fc["properties"] = {"backend": backend.name, "aoi": "St. Louis County demo"}
    Path(out_path).write_text(json.dumps(fc), encoding="utf-8")
    print(f"wrote {len(gdf)} footprints ({backend.name}) to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_stl.geojson")
