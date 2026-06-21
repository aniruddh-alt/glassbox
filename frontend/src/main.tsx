import "./instrument"; // MUST be first — Sentry initializes before any app code runs

import React from "react";
import { createRoot } from "react-dom/client";
import * as Sentry from "@sentry/react";

import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Sentry.ErrorBoundary fallback={<p>Something went wrong</p>} showDialog>
      <App />
    </Sentry.ErrorBoundary>
  </React.StrictMode>,
);
