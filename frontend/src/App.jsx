import { useCallback, useState } from "react";
import { listJobs } from "./api.js";
import MapView from "./components/MapView.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { useJob } from "./useJob.js";

export default function App() {
  const [drawMode, setDrawMode] = useState(false);
  const [aoi, setAoi] = useState(null);
  const [basemap, setBasemap] = useState("imagery");
  const [demoError, setDemoError] = useState(null);
  const { job, buildings, error, submit, watch, reset } = useJob();

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
        onLoadDemo={onLoadDemo}
        basemap={basemap}
        onBasemap={setBasemap}
      />
      <MapView
        basemap={basemap}
        drawMode={drawMode}
        aoi={aoi}
        buildings={buildings}
        onBBoxDrawn={onBBoxDrawn}
      />
    </div>
  );
}
