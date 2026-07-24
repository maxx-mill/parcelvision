import { useCallback, useEffect, useState } from "react";
import { listJobs } from "./api.js";
import MapView from "./components/MapView.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { useJob } from "./useJob.js";
import { useValidation } from "./useValidation.js";

export default function App() {
  const [drawMode, setDrawMode] = useState(false);
  const [aoi, setAoi] = useState(null);
  const [basemap, setBasemap] = useState("imagery");
  const [demoError, setDemoError] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);
  const { job, buildings, error, submit, watch, cancel, reset } = useJob();
  const validation = useValidation(job);

  // Refresh history when the active job settles (and on load).
  const jobStatus = job?.status;
  useEffect(() => {
    listJobs().then(setRecentJobs).catch(() => {});
  }, [jobStatus]);

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
      />
      <MapView
        basemap={basemap}
        drawMode={drawMode}
        aoi={aoi}
        buildings={buildings}
        parcels={validation.result?.parcels}
        onBBoxDrawn={onBBoxDrawn}
      />
    </div>
  );
}
