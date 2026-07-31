"""Chapter 7 — parcel lookup + per-parcel property report."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import parcels as lookup
from ..db import get_session
from ..models import Job

router = APIRouter(tags=["parcels"])


@router.get("/parcels/search")
def search_parcels(q: str = Query(min_length=3), limit: int = 10) -> dict:
    """Find county parcels whose situs address contains `q`."""
    try:
        results = lookup.search_by_address(q, limit)
    except Exception as exc:  # upstream county service hiccup
        raise HTTPException(502, detail=f"parcel service error: {exc}") from exc
    return {"parcels": results}


@router.get("/parcels/at")
def parcel_at_point(lon: float, lat: float) -> dict:
    """The parcel containing a clicked point."""
    try:
        parcel = lookup.parcel_at(lon, lat)
    except Exception as exc:
        raise HTTPException(502, detail=f"parcel service error: {exc}") from exc
    if parcel is None:
        raise HTTPException(404, detail="no parcel at that location")
    return parcel


# Per-parcel property report: the structures whose interior point falls on the
# parcel, with roof condition + areas, plus a summary. Reads the parcel geometry
# from the parcels table (loaded when the job was validated).
_REPORT_SQL = """
WITH p AS (SELECT geom, address FROM parcels WHERE locator = :locator),
jb AS (
    SELECT area_sqm, condition, tarp_fraction, heterogeneity, geom,
           ST_PointOnSurface(geom) AS pt
    FROM buildings WHERE job_id = :jid
)
SELECT jb.area_sqm, jb.condition, jb.tarp_fraction, jb.heterogeneity,
       ST_Overlaps((SELECT geom FROM p), jb.geom) AS crosses
FROM jb WHERE ST_Contains((SELECT geom FROM p), jb.pt)
"""


# Condition severity mirrors the classifier's flag(): tarp > damaged > review > ok.
# "damaged" was added with the v5 model; leaving it out here made a collapsed
# roof report "ok" as the worst condition.
_SEVERITY = ("tarp", "damaged", "review", "ok")


def summarize_structures(structures: list[dict]) -> dict:
    """Roll a parcel's structures into count/area/condition summary fields."""
    cond_counts = {c: 0 for c in ("ok", "review", "damaged", "tarp")}
    for st in structures:
        key = st.get("condition") or "ok"
        cond_counts[key] = cond_counts.get(key, 0) + 1
    worst = next((c for c in _SEVERITY if cond_counts.get(c)), "ok")
    return {
        "structure_count": len(structures),
        "total_building_area_sqm": round(sum(st.get("area_sqm") or 0 for st in structures), 1),
        "condition_counts": cond_counts,
        "worst_condition": worst,
    }


@router.get("/jobs/{job_id}/report")
def property_report(
    job_id: uuid.UUID, locator: str, session: Session = Depends(get_session)
) -> dict:
    """Structures on one parcel with condition + a summary (the property report)."""
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, detail="job not found")
    parcel = session.execute(
        text("SELECT locator, address FROM parcels WHERE locator = :loc"), {"loc": locator}
    ).one_or_none()
    if parcel is None:
        raise HTTPException(409, detail="parcel not loaded — run validation for this job first")

    rows = session.execute(text(_REPORT_SQL), {"jid": str(job_id), "locator": locator}).all()
    structures = [
        {
            "area_sqm": r.area_sqm,
            "condition": r.condition,
            "tarp_fraction": r.tarp_fraction,
            "heterogeneity": r.heterogeneity,
            "crosses_boundary": bool(r.crosses),
        }
        for r in rows
    ]

    return {
        "parcel": {"locator": parcel.locator, "address": parcel.address},
        "structures": structures,
        "summary": {
            **summarize_structures(structures),
            "any_crossing": any(st["crosses_boundary"] for st in structures),
        },
    }
