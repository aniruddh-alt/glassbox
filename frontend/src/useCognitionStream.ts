// Consume POST /api/chat as NDJSON via fetch + ReadableStream (NOT EventSource — GET-only,
// and SSE-over-GET breaks through Cloudflare). Handles split-UTF8 chunk boundaries.
// OWNER: Lane C. Build against fixtures/cognition_event.sample.json before the API is real.
import { useState } from "react";

import type { CognitionEvent, StreamLine } from "./types";

export function useCognitionStream() {
  const [answer, setAnswer] = useState("");
  const [event, setEvent] = useState<CognitionEvent | null>(null);
  const [status, setStatus] = useState<"idle" | "streaming" | "done" | "error">("idle");

  async function send(messages: { role: string; content: string }[], trackers?: string[]) {
    setAnswer("");
    setEvent(null);
    setStatus("streaming");
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
        buf = lines.pop() ?? ""; // keep partial line
        for (const line of lines) {
          if (!line.trim()) continue;
          const msg = JSON.parse(line) as StreamLine;
          if (msg.type === "token") setAnswer((a) => a + msg.text);
          else setEvent(msg); // the final {type:"event", ...CognitionEvent}
        }
      }
      setStatus("done");
    } catch {
      setStatus("error");
    }
  }

  return { answer, event, status, send };
}
