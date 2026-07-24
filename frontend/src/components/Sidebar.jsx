import { exportUrl } from "../api.js";

const STAGES = [
  ["queued", "Queued"],
  ["fetching_imagery", "Fetching NAIP imagery"],
  ["running_inference", "Running segmentation"],
  ["vectorizing", "Vectorizing + regularizing"],
  ["writing_db", "Writing to PostGIS"],
];
const EXPORTS = [
  ["geojson", "GeoJSON"],
  ["gpkg", "GeoPackage"],
  ["shp", "Shapefile"],
  ["fgdb", "File GDB"],
];

function bboxAreaKm2([minLon, minLat, maxLon, maxLat]) {
  const latKm = 110.574;
  const lonKm = 111.32 * Math.cos((((minLat + maxLat) / 2) * Math.PI) / 180);
  return (maxLat - minLat) * latKm * (maxLon - minLon) * lonKm;
}

function StageList({ status }) {
  if (status === "canceled") return <p className="meta muted">Canceled.</p>;
  const idx = STAGES.findIndex(([key]) => key === status);
  const doneAll = status === "done";
  return (
    <ol className="stages">
      {STAGES.map(([key, label], i) => {
        const cls = doneAll || i < idx ? "done" : i === idx ? "active" : "";
        return (
          <li key={key} className={cls}>
            <span className="dot" />
            {label}
          </li>
        );
      })}
    </ol>
  );
}

function RecentJobs({ jobs, activeId, onSelect }) {
  if (!jobs.length) return null;
  return (
    <section>
      <h2>History</h2>
      <ul className="joblist">
        {jobs.slice(0, 8).map((j) => (
          <li key={j.id}>
            <button
              className={j.id === activeId ? "jobrow on" : "jobrow"}
              disabled={j.status !== "done"}
              onClick={() => onSelect(j)}
              title={j.status === "done" ? "Show these results" : j.status}
            >
              <span className={`st st-${j.status}`} />
              <span className="jid">{j.id.slice(0, 8)}</span>
              <span className="jmeta">
                {j.status === "done" ? `${j.building_count} bldgs` : j.status}
                {j.is_seed ? " · demo" : ""}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

const VAL_STAGE_LABEL = {
  loading_parcels: "Loading county parcels…",
  validating: "Cross-checking footprints…",
};

function ValidationPanel({ job, validation }) {
  const { status, result, error, validate } = validation;
  const busy = status === "loading_parcels" || status === "validating";

  return (
    <div className="validation">
      <h3>Parcel validation</h3>
      {status == null && (
        <>
          <p className="meta muted">
            Check detected footprints against authoritative county parcels.
          </p>
          <button className="btn small" onClick={validate}>
            Validate against parcels
          </button>
        </>
      )}
      {busy && <p className="meta">{VAL_STAGE_LABEL[status]}</p>}
      {status === "failed" && <p className="error">{error || "validation failed"}</p>}
      {status === "done" && result && (
        <>
          <div className="valstats">
            <div className="valstat">
              <span className="num">{result.summary.parcels_total}</span>parcels
            </div>
            <div className="valstat">
              <span className="num red">{result.summary.parcels_empty}</span>no structure
            </div>
            <div className="valstat">
              <span className="num amber">{result.summary.buildings_crossing}</span>cross a line
            </div>
            <div className="valstat">
              <span className="num">{result.summary.buildings_off_parcel}</span>off-parcel
            </div>
          </div>
          <ul className="vallegend">
            <li>
              <span className="sw" style={{ background: "#ef4444" }} /> parcel, no detected structure
            </li>
            <li>
              <span className="sw" style={{ background: "#f59e0b" }} /> parcel, multiple detections
            </li>
            <li>
              <span className="sw" style={{ background: "#22c55e" }} /> parcel, one detection
            </li>
            <li>
              <span className="sw line" style={{ background: "#f43f5e" }} /> footprint crosses a boundary
            </li>
          </ul>
          <p className="fineprint">
            Parcels: St. Louis County open data. CV can't see cadastral lines — this join is how you
            catch detection and boundary errors.
          </p>
        </>
      )}
    </div>
  );
}

export default function Sidebar({
  drawMode,
  onToggleDraw,
  aoi,
  job,
  buildings,
  error,
  onExtract,
  onCancel,
  recentJobs,
  onSelectJob,
  onLoadDemo,
  basemap,
  onBasemap,
  validation,
}) {
  const running = job && job.status !== "done" && job.status !== "failed";
  const area = aoi ? bboxAreaKm2(aoi) : null;

  return (
    <aside className="sidebar">
      <header>
        <div className="brand">
          <span className="mark">▣</span>
          <h1>ParcelVision</h1>
        </div>
        <p className="tagline">
          Building footprints from NAIP aerial imagery — extracted with CV, regularized, stored in
          PostGIS, exportable.
        </p>
      </header>

      <section>
        <h2>1 · Area of interest</h2>
        <button className={drawMode ? "btn active" : "btn"} onClick={onToggleDraw} disabled={running}>
          {drawMode ? "Drag a rectangle on the map…" : aoi ? "Redraw area" : "Draw area"}
        </button>
        {aoi && (
          <p className="meta">
            {aoi.map((v) => v.toFixed(4)).join(", ")}
            <br />≈ {area.toFixed(2)} km²{area > 1 && " — over the 1 km² demo cap"}
          </p>
        )}
      </section>

      <section>
        <h2>2 · Extract</h2>
        <button className="btn primary" onClick={onExtract} disabled={!aoi || running}>
          {running ? "Working…" : "Extract buildings"}
        </button>
        {running && (
          <button className="btn ghost cancel" onClick={onCancel}>
            Cancel job
          </button>
        )}
        {job && <StageList status={job.status} />}
        {job?.status === "failed" && <p className="error">{job.error}</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <section>
        <h2>3 · Results</h2>
        {job?.status === "done" ? (
          <>
            <p className="meta">
              <strong>{job.building_count}</strong> footprints · backend {job.backend}
              {job.is_seed && " · precomputed demo"}
            </p>
            <div className="exports">
              {EXPORTS.map(([fmt, label]) => (
                <a key={fmt} className="btn small" href={exportUrl(job.id, fmt)}>
                  {label}
                </a>
              ))}
            </div>
            <ValidationPanel job={job} validation={validation} />
          </>
        ) : (
          <p className="meta muted">
            {buildings ? "" : "Run an extraction, or load the precomputed demo below."}
          </p>
        )}
      </section>

      <RecentJobs jobs={recentJobs} activeId={job?.id} onSelect={onSelectJob} />

      <footer>
        <button className="btn ghost" onClick={onLoadDemo} disabled={running}>
          Load demo AOI (no inference)
        </button>
        <div className="basemap-toggle">
          <label className={basemap === "streets" ? "on" : ""}>
            <input
              type="radio"
              name="basemap"
              checked={basemap === "streets"}
              onChange={() => onBasemap("streets")}
            />
            Streets
          </label>
          <label className={basemap === "imagery" ? "on" : ""}>
            <input
              type="radio"
              name="basemap"
              checked={basemap === "imagery"}
              onChange={() => onBasemap("imagery")}
            />
            Imagery
          </label>
        </div>
        <p className="fineprint">
          CV finds what's visible — buildings, not legal parcel lines. Validate a finished job
          against county parcels to catch detection and boundary errors.
        </p>
      </footer>
    </aside>
  );
}
