import "./instrument"; // MUST be first — Sentry initializes before any app code runs

import React from "react";
import { createRoot } from "react-dom/client";
import * as Sentry from "@sentry/react";

// Self-hosted type (offline, no runtime requests). One grotesque throughout.
import "@fontsource/hanken-grotesk/400.css";
import "@fontsource/hanken-grotesk/500.css";
import "@fontsource/hanken-grotesk/600.css";

import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Sentry.ErrorBoundary fallback={<p>Something went wrong</p>} showDialog>
      <App />
    </Sentry.ErrorBoundary>
  </React.StrictMode>,
);
