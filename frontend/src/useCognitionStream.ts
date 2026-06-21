// Consume POST /api/chat as NDJSON via fetch + ReadableStream (NOT EventSource — GET-only,
// and SSE-over-GET breaks through Cloudflare). Handles split-UTF8 chunk boundaries.
// OWNER: Lane C. Pass {mock:true} to replay fixtures/the demo event before the API is real.
import { useState } from "react";

import type { CognitionEvent, StreamLine } from "./types";
import { DEMO_EVENT } from "./mock";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function useCognitionStream(opts?: { mock?: boolean }) {
  const [answer, setAnswer] = useState("");
  const [event, setEvent] = useState<CognitionEvent | null>(null);
  const [status, setStatus] = useState<"idle" | "streaming" | "done" | "error">("idle");

  // Mock mode: replay the demo event token-by-token (post-hoc: tokens stream, then the event lands).
  async function replayMock() {
    setAnswer(""); setEvent(null); setStatus("streaming");
    for (const tok of DEMO_EVENT.io.response.split(/(\s+)/)) {
      setAnswer((a) => a + tok);
      await sleep(26 + Math.random() * 40);
    }
    await sleep(500);
    setEvent(DEMO_EVENT); setStatus("done");
  }

  async function send(messages: { role: string; content: string }[], trackers?: string[]) {
    if (opts?.mock) return replayMock();
    setAnswer(""); setEvent(null); setStatus("streaming");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, trackers }),
      });
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line) as StreamLine;
          if (msg.type === "token") setAnswer((a) => a + msg.text);
          else setEvent(msg);
        }
      }
      setStatus("done");
    } catch {
      setStatus("error");
    }
  }

  return { answer, event, status, send };
}
