"""Load precomputed demo results into the DB (the `make demo` path).

Usage: python scripts/load_seed.py /seed/demo_stl.geojson

The GeoJSON is a FeatureCollection of building footprints produced by a real
pipeline run (scripts/generate_seed.py in worker/), with per-feature
`confidence` and `area_sqm` properties and a top-level `bbox`. Re-running
replaces any previous seed job.
"""

import json
import sys
import uuid
from pathlib import Path

from sqlalchemy import delete, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import get_engine, init_db  # noqa: E402
from app.models import Building, Job  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402


def main(path: str) -> None:
    fc = json.loads(Path(path).read_text(encoding="utf-8"))
    features = fc["features"]
    if not features:
        raise SystemExit(f"{path} contains no features")
    bbox = fc.get("bbox")
    if bbox is None:
        raise SystemExit(f"{path} is missing a top-level bbox")

    init_db()
    with Session(get_engine()) as session:
        old_ids = [j.id for j in session.query(Job).filter(Job.is_seed).all()]
        if old_ids:
            session.execute(delete(Job).where(Job.id.in_(old_ids)))

        job = Job(
            id=uuid.uuid4(),
            status="done",
            bbox=list(bbox),
            backend=fc.get("properties", {}).get("backend", "local_cpu"),
            building_count=len(features),
            is_seed=True,
        )
        session.add(job)
        session.flush()
        for f in features:
            props = f.get("properties") or {}
            session.add(
                Building(
                    job_id=job.id,
                    geom=text("ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326)").bindparams(
                        g=json.dumps(f["geometry"])
                    ),
                    confidence=props.get("confidence"),
                    area_sqm=props.get("area_sqm"),
                    condition=props.get("condition"),
                    roof_damage_score=props.get("roof_damage_score"),
                    tarp_fraction=props.get("tarp_fraction"),
                )
            )
        session.commit()
        print(f"Seeded job {job.id} with {len(features)} buildings (replaced {len(old_ids)} old)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
