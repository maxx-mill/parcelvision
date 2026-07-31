import { useCallback, useEffect, useState } from "react";
import { getReport, listJobs, searchParcels } from "./api.js";
import MapView from "./components/MapView.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { useJob } from "./useJob.js";
import { useValidation } from "./useValidation.js";

// Expand a parcel bbox by a margin (≈40% of span, min ~45 m) for analysis.
function bufferBbox([minx, miny, maxx, maxy]) {
  const mx = Math.max((maxx - minx) * 0.4, 0.0004);
  const my = Math.max((maxy - miny) * 0.4, 0.0004);
  return [minx - mx, miny - my, maxx + mx, maxy + my];
}

export default function App() {
  const [drawMode, setDrawMode] = useState(false);
  const [aoi, setAoi] = useState(null);
  const [basemap, setBasemap] = useState("imagery");
  const [demoError, setDemoError] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);
  // Chapter 7 — parcel-first workflow
  const [parcelResults, setParcelResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [parcel, setParcel] = useState(null); // { locator, address, bbox, geometry }
  const [report, setReport] = useState(null);
  const { job, buildings, error, submit, watch, cancel, reset } = useJob();
  const validation = useValidation(job);

  // Refresh history when the active job settles (and on load).
  const jobStatus = job?.status;
  useEffect(() => {
    listJobs().then(setRecentJobs).catch(() => {});
  }, [jobStatus]);

  // Parcel workflow auto-chain: analyze -> job done -> validate -> report.
  const valStatus = validation.status;
  useEffect(() => {
    if (parcel && jobStatus === "done" && valStatus === null) validation.validate();
  }, [parcel, jobStatus, valStatus, validation]);
  useEffect(() => {
    if (parcel && valStatus === "done" && job) {
      getReport(job.id, parcel.locator).then(setReport).catch(() => setReport(null));
    }
  }, [parcel, valStatus, job]);

  const onParcelSearch = useCallback(async (q) => {
    setDemoError(null);
    setSearching(true);
    try {
      const { parcels } = await searchParcels(q);
      setParcelResults(parcels);
    } catch (e) {
      setDemoError(e.message);
    } finally {
      setSearching(false);
    }
  }, []);

  const onPickParcel = useCallback(
    (p) => {
      reset();
      setReport(null);
      setParcel(p);
      // Analyze the parcel plus a small margin so edge structures are fully
      // captured and the imagery isn't microscopic; the report still filters
      // to structures whose interior point falls on the parcel itself.
      setAoi(bufferBbox(p.bbox));
      setParcelResults(null);
      setDrawMode(false);
    },
    [reset]
  );

  const onSelectJob = useCallback(
    (j) => {
      setAoi(j.bbox);
      watch(j);
    },
    [watch]
  );

  const onBBoxDrawn = useCallback((bbox) => {
    setAoi(bbox);
    setDrawMode(false);
  }, []);

  const onToggleDraw = () => {
    reset();
    setParcel(null);
    setReport(null);
    setAoi(null);
    setDrawMode((d) => !d);
  };

  const onExtract = () => aoi && submit(aoi);

  const onLoadDemo = async () => {
    setDemoError(null);
    try {
      const jobs = await listJobs();
      const seed = jobs.find((j) => j.is_seed && j.status === "done");
      if (!seed) {
        setDemoError("No seed data loaded — run `make demo` first.");
        return;
      }
      setAoi(seed.bbox);
      watch(seed);
    } catch (e) {
      setDemoError(e.message);
    }
  };

  return (
    <div className="app">
      <Sidebar
        drawMode={drawMode}
        onToggleDraw={onToggleDraw}
        aoi={aoi}
        job={job}
        buildings={buildings}
        error={error ?? demoError}
        onExtract={onExtract}
        onCancel={cancel}
        recentJobs={recentJobs}
        onSelectJob={onSelectJob}
        onLoadDemo={onLoadDemo}
        basemap={basemap}
        onBasemap={setBasemap}
        validation={validation}
        onParcelSearch={onParcelSearch}
        parcelResults={parcelResults}
        searching={searching}
        onPickParcel={onPickParcel}
        parcel={parcel}
        report={report}
      />
      <MapView
        basemap={basemap}
        drawMode={drawMode}
        aoi={aoi}
        buildings={buildings}
        parcels={validation.result?.parcels}
        selectedParcel={parcel?.geometry}
        onBBoxDrawn={onBBoxDrawn}
      />
    </div>
  );
}
