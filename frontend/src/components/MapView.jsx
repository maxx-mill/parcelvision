// maplibre-gl 6 has no default export
import { LngLatBounds, Map as MaplibreMap, NavigationControl, Popup } from "maplibre-gl";
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

// On-map legend for footprint colours — condition when the classifier ran,
// else a single "detected building" swatch. Mirrors the paint match below.
function MapLegend({ buildings }) {
  const feats = buildings?.features;
  if (!feats?.length) return null;
  const hasCondition = feats.some((f) => f.properties?.condition);
  return (
    <div className="map-legend" aria-hidden="true">
      <p className="legend-title">Footprints</p>
      <ul>
        {hasCondition ? (
          <>
            <li><span className="sw" style={{ background: "#22d3ee" }} /> intact roof</li>
            <li><span className="sw" style={{ background: "#f59e0b" }} /> review</li>
            <li><span className="sw" style={{ background: "#dc2626" }} /> likely damage</li>
            <li><span className="sw" style={{ background: "#a855f7" }} /> tarp</li>
          </>
        ) : (
          <li><span className="sw" style={{ background: "#22d3ee" }} /> detected building</li>
        )}
      </ul>
    </div>
  );
}

// Parcel fill by validation flag: red = no detected structure (the headline
// finding), amber = more than one detection, faint green = a clean 1:1 match.
const PARCEL_FILL = [
  "match",
  ["get", "flag"],
  "empty", "#ef4444",
  "multi", "#f59e0b",
  "#22c55e",
];

export default function MapView({
  basemap,
  drawMode,
  aoi,
  buildings,
  parcels,
  selectedParcel,
  onBBoxDrawn,
}) {
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
      map.addSource("parcels", { type: "geojson", data: EMPTY });
      map.addSource("selparcel", { type: "geojson", data: EMPTY });
      map.addSource("buildings", { type: "geojson", data: EMPTY });

      // The parcel the user picked (Chapter 7) — a bright highlight under all.
      map.addLayer({
        id: "selparcel-fill",
        type: "fill",
        source: "selparcel",
        paint: { "fill-color": "#a855f7", "fill-opacity": 0.12 },
      });
      map.addLayer({
        id: "selparcel-line",
        type: "line",
        source: "selparcel",
        paint: { "line-color": "#a855f7", "line-width": 2.5 },
      });

      // Parcels sit UNDER the detected footprints so both read at once.
      map.addLayer({
        id: "parcels-fill",
        type: "fill",
        source: "parcels",
        paint: { "fill-color": PARCEL_FILL, "fill-opacity": 0.25 },
      });
      map.addLayer({
        id: "parcels-line",
        type: "line",
        source: "parcels",
        paint: {
          // Parcels a footprint crosses get a bright boundary.
          "line-color": ["case", [">", ["get", "crossing_count"], 0], "#f43f5e", "#64748b"],
          "line-width": ["case", [">", ["get", "crossing_count"], 0], 1.6, 0.6],
        },
      });

      // Colour footprints by roof-condition indicator: tarp (red) > review
      // (amber) > ok (cyan). Falls back to cyan when condition is absent.
      const conditionColor = [
        "match",
        ["get", "condition"],
        "damaged",
        "#dc2626",
        "tarp",
        "#a855f7",
        "review",
        "#f59e0b",
        "#22d3ee",
      ];
      map.addLayer({
        id: "buildings-fill",
        type: "fill",
        source: "buildings",
        paint: { "fill-color": conditionColor, "fill-opacity": 0.4 },
      });
      map.addLayer({
        id: "buildings-line",
        type: "line",
        source: "buildings",
        paint: { "line-color": conditionColor, "line-width": 1.5 },
      });
      map.addLayer({
        id: "aoi-line",
        type: "line",
        source: "aoi",
        paint: { "line-color": "#f59e0b", "line-width": 2, "line-dasharray": [2, 2] },
      });

      // Click a footprint to inspect it before exporting.
      map.on("click", "buildings-fill", (e) => {
        const p = e.features[0]?.properties ?? {};
        const conf = p.confidence != null ? Number(p.confidence).toFixed(2) : "—";
        const area = p.area_sqm != null ? `${Number(p.area_sqm).toLocaleString()} m²` : "—";
        const labels = {
          damaged: "⚠ likely roof damage",
          tarp: "⚠ tarp detected",
          review: "review",
          ok: "roof looks intact",
        };
        let cond = "";
        if (p.condition) {
          const dmg = p.roof_damage_score != null ? `damage score ${Number(p.roof_damage_score).toFixed(2)}` : "";
          const tarp = p.tarp_fraction > 0.02 ? `${(p.tarp_fraction * 100).toFixed(0)}% tarp` : "";
          cond =
            `<br/><span class="cond cond-${p.condition}">${labels[p.condition] ?? p.condition}</span>` +
            `<br/><span class="muted">${[dmg, tarp].filter(Boolean).join(" · ")}</span>`;
        }
        new Popup({ closeButton: false, maxWidth: "240px" })
          .setLngLat(e.lngLat)
          .setHTML(
            `<div class="bldg-popup"><strong>Detected building</strong>` +
              `<br/>confidence ${conf} · ${area}${cond}</div>`
          )
          .addTo(map);
      });
      map.on("mouseenter", "buildings-fill", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "buildings-fill", () => {
        map.getCanvas().style.cursor = "";
      });

      // Click a parcel (where no footprint covers it) to see its validation.
      map.on("click", "parcels-fill", (e) => {
        if (map.queryRenderedFeatures(e.point, { layers: ["buildings-fill"] }).length) return;
        const p = e.features[0]?.properties ?? {};
        const verdict =
          p.building_count === 0
            ? "no detected structure"
            : `${p.building_count} detected structure${p.building_count > 1 ? "s" : ""}`;
        const crossing = p.crossing_count > 0 ? `<br/>${p.crossing_count} footprint(s) cross the line` : "";
        new Popup({ closeButton: false, maxWidth: "240px" })
          .setLngLat(e.lngLat)
          .setHTML(
            `<div class="bldg-popup"><strong>Parcel ${p.locator ?? ""}</strong>` +
              `<br/>${p.address ?? ""}<br/>${verdict}${crossing}</div>`
          )
          .addTo(map);
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

  // Selected parcel highlight (Chapter 7) + zoom to it.
  useEffect(() => {
    const map = mapRef.current;
    whenReady(readyRef, map, () => {
      if (!selectedParcel) {
        map.getSource("selparcel").setData(EMPTY);
        return;
      }
      map.getSource("selparcel").setData({ type: "Feature", geometry: selectedParcel });
      const b = new LngLatBounds();
      const rings =
        selectedParcel.type === "Polygon" ? [selectedParcel.coordinates] : selectedParcel.coordinates;
      for (const poly of rings) for (const pt of poly[0]) b.extend(pt);
      map.fitBounds(b, { padding: 120, duration: 600, maxZoom: 19 });
    });
  }, [selectedParcel]);

  // Parcel validation layer
  useEffect(() => {
    const map = mapRef.current;
    whenReady(readyRef, map, () => {
      map.getSource("parcels").setData(parcels ?? EMPTY);
    });
  }, [parcels]);

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

  return (
    <div className="map-wrap">
      <div ref={containerRef} className="map" />
      <MapLegend buildings={buildings} />
    </div>
  );
}
