import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // Local dev outside Docker: talk to the api container's published port.
    proxy: { "/api": "http://localhost:8000" },
  },
});
