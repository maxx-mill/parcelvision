// maplibre-gl 6 has no default export
import { LngLatBounds, Map as MaplibreMap, NavigationControl } from "maplibre-gl";
import { useEffect, useRef } from "react";

const EMPTY = { type: "FeatureCollection", features: [] };

// Both basemaps live in one style; switching toggles layer visibility so our
// overlay sources survive (setStyle would wipe them).
const BASE_STYLE = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
    },
    esri: {
      type: "raster",
      tiles: [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      ],
      tileSize: 256,
      attribution: "Imagery © Esri & contributors",
    },
  },
  layers: [
    { id: "basemap-streets", type: "raster", source: "osm" },
    { id: "basemap-imagery", type: "raster", source: "esri", layout: { visibility: "none" } },
  ],
};

// Run now if the style is loaded, else once it is — early prop updates
// (e.g. demo results arriving before `load` fires) must not be dropped.
function whenReady(readyRef, map, fn) {
  if (readyRef.current) fn();
  else map?.once("load", fn);
}

function bboxToPolygon([minx, miny, maxx, maxy]) {
  return {
    type: "Feature",
    geometry: {
      type: "Polygon",
      coordinates: [
        [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]],
      ],
    },
  };
}

export default function MapView({ basemap, drawMode, aoi, buildings, onBBoxDrawn }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const readyRef = useRef(false);

  useEffect(() => {
    const map = new MaplibreMap({
      container: containerRef.current,
      style: BASE_STYLE,
      center: [-90.32, 38.64], // St. Louis County
      zoom: 12,
      attributionControl: { compact: true },
    });
    map.addControl(new NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      map.addSource("aoi", { type: "geojson", data: EMPTY });
      map.addSource("buildings", { type: "geojson", data: EMPTY });
      map.addLayer({
        id: "buildings-fill",
        type: "fill",
        source: "buildings",
        paint: { "fill-color": "#22d3ee", "fill-opacity": 0.35 },
      });
      map.addLayer({
        id: "buildings-line",
        type: "line",
        source: "buildings",
        paint: { "line-color": "#06b6d4", "line-width": 1.5 },
      });
      map.addLayer({
        id: "aoi-line",
        type: "line",
        source: "aoi",
        paint: { "line-color": "#f59e0b", "line-width": 2, "line-dasharray": [2, 2] },
      });
      readyRef.current = true;
    });
    mapRef.current = map;
    return () => map.remove();
  }, []);

  // Basemap toggle
  useEffect(() => {
    const map = mapRef.current;
    whenReady(readyRef, map, () => {
      map.setLayoutProperty("basemap-streets", "visibility", basemap === "streets" ? "visible" : "none");
      map.setLayoutProperty("basemap-imagery", "visibility", basemap === "imagery" ? "visible" : "none");
    });
  }, [basemap]);

  // AOI rectangle
  useEffect(() => {
    const map = mapRef.current;
    whenReady(readyRef, map, () => {
      map.getSource("aoi").setData(aoi ? bboxToPolygon(aoi) : EMPTY);
    });
  }, [aoi]);

  // Results layer
  useEffect(() => {
    const map = mapRef.current;
    whenReady(readyRef, map, () => {
      map.getSource("buildings").setData(buildings ?? EMPTY);
      if (buildings?.features?.length) {
        const bounds = new LngLatBounds();
        for (const f of buildings.features) {
          const rings = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates;
          for (const ring of rings) for (const pt of ring[0]) bounds.extend(pt);
        }
        map.fitBounds(bounds, { padding: 80, duration: 600 });
      }
    });
  }, [buildings]);

  // Rectangle draw interaction
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !drawMode) return;
    const canvas = map.getCanvas();
    map.dragPan.disable();
    canvas.style.cursor = "crosshair";
    let start = null;

    const rect = (a, b) => [
      Math.min(a.lng, b.lng), Math.min(a.lat, b.lat),
      Math.max(a.lng, b.lng), Math.max(a.lat, b.lat),
    ];
    const onDown = (e) => {
      start = e.lngLat;
      e.preventDefault();
    };
    const onMove = (e) => {
      if (start) map.getSource("aoi").setData(bboxToPolygon(rect(start, e.lngLat)));
    };
    const onUp = (e) => {
      if (!start) return;
      const bbox = rect(start, e.lngLat);
      start = null;
      onBBoxDrawn(bbox);
    };
    map.on("mousedown", onDown);
    map.on("mousemove", onMove);
    map.on("mouseup", onUp);
    return () => {
      map.off("mousedown", onDown);
      map.off("mousemove", onMove);
      map.off("mouseup", onUp);
      map.dragPan.enable();
      canvas.style.cursor = "";
    };
  }, [drawMode, onBBoxDrawn]);

  return <div ref={containerRef} className="map" />;
}
