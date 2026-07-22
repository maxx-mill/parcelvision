# County parcel data (Chapter 3)

Authoritative parcel polygons for validating detected footprints. Planned
source: **St. Louis County Open Government / GIS open data** parcel layer
(with St. Louis City's open data portal as the in-city complement).

Loader lands in Chapter 3:

1. Bulk-download the parcel layer (GeoJSON/shapefile export or ArcGIS REST
   `/query` pagination).
2. `to_postgis` into a `parcels` table (EPSG:4326, `parcel_id`, owner-neutral
   attributes only).
3. Spatial-join validation views: buildings-per-parcel, footprints crossing
   parcel lines, parcels with no detected structure.

Until then this directory is a stub so the MVP doesn't block on parcel
sourcing (assumption A5 in the build brief).
