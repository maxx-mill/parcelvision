import { setWorkerUrl } from "maplibre-gl";
// maplibre 6 loads its render worker relative to the bundle URL, which breaks
// under Vite's hashed asset output; ?worker&url makes Vite bundle the worker
// (including its shared chunk import) and hand back a servable URL.
import maplibreWorkerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "maplibre-gl/dist/maplibre-gl.css";
import "./styles.css";

setWorkerUrl(maplibreWorkerUrl);

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
