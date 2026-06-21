// Sentry must initialize BEFORE any other app code — this file is imported first in main.tsx.
// Mirrors the backend's PHI-careful posture (see backend/fanout.py): replay masks all text,
// and only error sessions are recorded. The DSN comes from .env.local (gitignored).
import * as Sentry from "@sentry/react";

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN, // unset → SDK no-ops, app still runs
  environment: import.meta.env.MODE,
  release: import.meta.env.VITE_APP_VERSION,

  integrations: [
    Sentry.browserTracingIntegration(),
    Sentry.replayIntegration({
      maskAllText: true, // medical tool: never capture raw text into a replay
      blockAllMedia: true,
    }),
  ],

  // Tracing. Propagate trace headers to the same-origin /api calls (Vite proxies to :8000
  // in dev). The backend routes its OWN tracing to Phoenix (traces_sample_rate=0.0), so the
  // browser side shows in Sentry while backend spans live in Phoenix — by design.
  tracesSampleRate: 1.0,
  tracePropagationTargets: ["localhost", /^\/api\//],

  // Session Replay — PHI-safe: don't record normal sessions, only the ones around an error.
  replaysSessionSampleRate: 0.0,
  replaysOnErrorSampleRate: 1.0,

  enableLogs: true,
});
