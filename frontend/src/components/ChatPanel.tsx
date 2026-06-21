// Clinician chat: the question + the streamed model answer + the Run/Replay composer.
import { useEffect, useRef, useState } from "react";

export function ChatPanel({
  userMsg, answer, status, hasRun, onRun,
}: {
  userMsg: string;
  answer: string;
  status: "idle" | "streaming" | "done" | "error";
  hasRun: boolean;
  onRun: (content: string) => void;
}) {
  const [draft, setDraft] = useState(userMsg);
  const threadRef = useRef<HTMLDivElement>(null);
  const streaming = status === "streaming";

  useEffect(() => {
    const t = threadRef.current; if (t) t.scrollTop = t.scrollHeight;
  }, [answer]);

  return (
    <section className="chat glass">
      <div className="thread" ref={threadRef}>
        <div className="msg user">
          <span className="who">clinician</span>
          <div className="body">{userMsg}</div>
        </div>
        {(answer || streaming) && (
          <div className="msg bot">
            <span className="who">gemma-2-2b-it</span>
            <div className="body">{answer}{streaming && <span className="cursor" />}</div>
          </div>
        )}
      </div>
      <div className="composer">
        <textarea value={draft} onChange={(e) => setDraft(e.target.value)} />
        <button
          className={`send ${streaming ? "running" : ""}`}
          disabled={streaming}
          onClick={() => onRun(draft)}
        >
          {streaming ? "Running…" : hasRun ? "Replay ▸" : "Run ▸"}
        </button>
      </div>
    </section>
  );
}
