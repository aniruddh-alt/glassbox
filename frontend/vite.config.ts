import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { sentryVitePlugin } from "@sentry/vite-plugin";

// Source-map upload runs ONLY when SENTRY_AUTH_TOKEN is set (release/CI builds); without it
// the plugin is omitted so a plain `npm run build` never fails. Set SENTRY_AUTH_TOKEN +
// SENTRY_ORG + SENTRY_PROJECT to get readable (un-minified) stack traces in Sentry.
const sentryPlugins = process.env.SENTRY_AUTH_TOKEN
  ? [
      sentryVitePlugin({
        org: process.env.SENTRY_ORG,
        project: process.env.SENTRY_PROJECT,
        authToken: process.env.SENTRY_AUTH_TOKEN,
      }),
    ]
  : [];

// Proxies /api -> backend on :8000 so the SPA and API share an origin in dev.
export default defineConfig({
  build: { sourcemap: "hidden" }, // emit maps for upload; keep them out of the served bundle
  plugins: [react(), ...sentryPlugins],
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
});
