import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Proxies /api -> backend on :8000 so the SPA and API share an origin in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
