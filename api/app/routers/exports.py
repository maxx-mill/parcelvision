import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

import geopandas as gpd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from ..db import get_engine, get_session
from ..models import Job

router = APIRouter(prefix="/jobs", tags=["exports"])

# format -> (OGR driver, file extension, needs zip). Shapefile and FileGDB are
# multi-file formats, so they ship as .zip. OpenFileGDB write needs GDAL >= 3.6;
# the pyogrio wheel bundles a current GDAL.
FORMATS: dict[str, tuple[str, str, bool]] = {
    "geojson": ("GeoJSON", ".geojson", False),
    "gpkg": ("GPKG", ".gpkg", False),
    "shp": ("ESRI Shapefile", ".shp", True),
    "fgdb": ("OpenFileGDB", ".gdb", True),
}


@router.get("/{job_id}/export")
def export_buildings(
    job_id: uuid.UUID, format: str = "geojson", session: Session = Depends(get_session)
) -> FileResponse:
    if format not in FORMATS:
        raise HTTPException(400, detail=f"format must be one of {sorted(FORMATS)}")
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, detail="job not found")

    gdf = gpd.read_postgis(
        "SELECT geom, confidence, area_sqm FROM buildings WHERE job_id = %(job_id)s",
        con=get_engine(),
        geom_col="geom",
        params={"job_id": str(job_id)},
    )
    if gdf.empty:
        raise HTTPException(404, detail="no buildings for this job (yet)")
    gdf = gdf.set_crs(4326, allow_override=True)

    driver, ext, needs_zip = FORMATS[format]
    tmpdir = Path(tempfile.mkdtemp(prefix="pv_export_"))
    stem = f"buildings_{str(job_id)[:8]}"
    out_path = tmpdir / f"{stem}{ext}"
    try:
        # Shapefile can't hold a Polygon/MultiPolygon mix — promote for consistency.
        if format == "shp":
            gdf.geometry = gdf.geometry.apply(
                lambda g: g if g.geom_type == "MultiPolygon" else g.buffer(0)
            )
        gdf.to_file(out_path, driver=driver)
    except Exception as exc:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(500, detail=f"export failed: {exc}") from exc

    cleanup = BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True)
    if needs_zip:
        zip_path = tmpdir / f"{stem}_{format}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(tmpdir.rglob("*")):
                if p == zip_path or p.is_dir():
                    continue
                zf.write(p, p.relative_to(tmpdir))
        return FileResponse(
            zip_path, filename=zip_path.name, media_type="application/zip", background=cleanup
        )
    media = "application/geo+json" if format == "geojson" else "application/octet-stream"
    return FileResponse(out_path, filename=out_path.name, media_type=media, background=cleanup)
