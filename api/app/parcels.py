"""Chapter 7 — parcel lookup over the county ArcGIS REST service.

Thin, read-only proxy so the frontend can find a parcel by address or by a
clicked point and turn it into an analysis AOI. Geometry comes back as GeoJSON
in EPSG:4326 with a computed bbox.
"""

import requests
from shapely.geometry import shape

from .config import get_settings

REQUEST_TIMEOUT = 30


def _query(params: dict) -> list[dict]:
    s = get_settings()
    common = {
        "outFields": f"{s.parcel_locator_field},{s.parcel_address_field}",
        "outSR": "4326",
        "f": "geojson",
    }
    resp = requests.get(
        f"{s.parcel_service_url}/query", params={**common, **params}, timeout=REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    fc = resp.json()
    loc, addr = s.parcel_locator_field, s.parcel_address_field
    out = []
    for f in fc.get("features", []):
        geom = f.get("geometry")
        props = f.get("properties", {})
        if geom is None or props.get(loc) is None:
            continue
        out.append(
            {
                "locator": props.get(loc),
                "address": props.get(addr),
                "geometry": geom,
                "bbox": list(shape(geom).bounds),  # [minx, miny, maxx, maxy]
            }
        )
    return out


def search_by_address(q: str, limit: int = 10) -> list[dict]:
    s = get_settings()
    safe = q.replace("'", "''").upper()
    return _query(
        {
            "where": f"UPPER({s.parcel_address_field}) LIKE '%{safe}%'",
            "returnGeometry": "true",
            "resultRecordCount": min(limit, 25),
        }
    )[:limit]


def parcel_at(lon: float, lat: float) -> dict | None:
    hits = _query(
        {
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "returnGeometry": "true",
        }
    )
    return hits[0] if hits else None
