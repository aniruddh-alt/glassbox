// Clinician chat: a multi-turn thread + the streaming assistant turn + the composer.
import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";

export type Msg = { role: "user" | "assistant"; content: string };

function inlineMarkdown(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const token = m[0];
    const body = token.startsWith("**")
      ? token.slice(2, -2)
      : token.startsWith("*")
        ? token.slice(1, -1)
        : token.slice(1, -1);
    if (token.startsWith("**")) out.push(<strong key={m.index}>{body}</strong>);
    else if (token.startsWith("*")) out.push(<em key={m.index}>{body}</em>);
    else out.push(<code key={m.index}>{body}</code>);
    last = m.index + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function MarkdownText({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  const lines = text.split(/\r?\n/);
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }
    if (line.startsWith("```")) {
      const code: string[] = [];
      i += 1;
      while (i < lines.length && !lines[i].startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      i += lines[i]?.startsWith("```") ? 1 : 0;
      blocks.push(<pre key={i}><code>{code.join("\n")}</code></pre>);
      continue;
    }
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    if (heading) {
      const Tag = (`h${Math.min(heading[1].length + 2, 5)}`) as keyof JSX.IntrinsicElements;
      blocks.push(<Tag key={i}>{inlineMarkdown(heading[2])}</Tag>);
      i += 1;
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(<li key={i}>{inlineMarkdown(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>);
        i += 1;
      }
      blocks.push(<ul key={i}>{items}</ul>);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: ReactNode[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(<li key={i}>{inlineMarkdown(lines[i].replace(/^\s*\d+\.\s+/, ""))}</li>);
        i += 1;
      }
      blocks.push(<ol key={i}>{items}</ol>);
      continue;
    }

    const para: string[] = [line];
    i += 1;
    while (i < lines.length && lines[i].trim() && !/^(```|#{1,3}\s+|\s*[-*]\s+|\s*\d+\.\s+)/.test(lines[i])) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push(<p key={i}>{inlineMarkdown(para.join(" "))}</p>);
  }

  return <>{blocks.map((block, idx) => <Fragment key={idx}>{block}</Fragment>)}</>;
}

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
  const taRef = useRef<HTMLTextAreaElement>(null);
  const streaming = status === "streaming";

  function autosize() {
    const t = taRef.current; if (!t) return;
    t.style.height = "auto";
    t.style.height = `${Math.min(t.scrollHeight, 140)}px`;
  }
  useEffect(autosize, []); // fit the pre-filled question on mount

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
            <div className="body">{m.role === "assistant" ? <MarkdownText text={m.content} /> : m.content}</div>
          </div>
        ))}
        {streaming && (
          <div className="msg bot">
            <span className="who">{modelName}</span>
            <div className="body"><MarkdownText text={pending ?? ""} /><span className="cursor" /></div>
          </div>
        )}
      </div>
      <div className="composer">
        <textarea
          ref={taRef}
          value={draft}
          onChange={(e) => { setDraft(e.target.value); autosize(); }}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
        />
        <button className={`send ${streaming ? "running" : ""}`} disabled={streaming} onClick={submit}>
          {streaming ? "Running" : "Send"}
        </button>
      </div>
    </section>
  );
}
