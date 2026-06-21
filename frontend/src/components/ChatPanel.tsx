// Clinician chat: a multi-turn thread + the streaming assistant turn + the composer.
import { useEffect, useRef, useState } from "react";

export type Msg = { role: "user" | "assistant"; content: string };

export function ChatPanel({
  thread, pending, status, modelName, onSend,
}: {
  thread: Msg[];
  pending: string | null;            // the in-progress assistant text while streaming
  status: "idle" | "streaming" | "done" | "error";
  modelName: string;
  onSend: (content: string) => void;
}) {
  const [draft, setDraft] = useState("Is ibuprofen safe to take in the third trimester of pregnancy?");
  const threadRef = useRef<HTMLDivElement>(null);
  const streaming = status === "streaming";

  useEffect(() => {
    const t = threadRef.current;
    if (t) t.scrollTop = t.scrollHeight;
  }, [thread, pending]);

  function submit() {
    const content = draft.trim();
    if (!content || streaming) return;
    onSend(content);
    setDraft("");
  }

  const who = (role: Msg["role"]) => (role === "user" ? "clinician" : modelName);

  return (
    <section className="chat glass">
      <div className="thread" ref={threadRef}>
        {thread.length === 0 && !pending && (
          <div className="empty">Ask a clinical question to watch the model's features fire.</div>
        )}
        {thread.map((m, i) => (
          <div key={i} className={`msg ${m.role === "user" ? "user" : "bot"}`}>
            <span className="who">{who(m.role)}</span>
            <div className="body">{m.content}</div>
          </div>
        ))}
        {streaming && (
          <div className="msg bot">
            <span className="who">{modelName}</span>
            <div className="body">{pending}<span className="cursor" /></div>
          </div>
        )}
      </div>
      <div className="composer">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
        />
        <button className={`send ${streaming ? "running" : ""}`} disabled={streaming} onClick={submit}>
          {streaming ? "Running" : "Send"}
        </button>
      </div>
    </section>
  );
}
