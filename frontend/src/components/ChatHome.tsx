import { Fragment, useState, useRef, useLayoutEffect, type KeyboardEvent, type ReactNode } from "react";
import "./ChatHome.css";

interface SourceCitation {
  number: number;
  title: string;
  url: string | null;
  chunk_id: string;
}

interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  ts: Date;
  sources?: SourceCitation[];
  error?: boolean;
}

let _id = 0;
const uid = () => ++_id;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isSourceCitation(value: unknown): value is SourceCitation {
  if (!isRecord(value)) return false;
  return (
    typeof value.number === "number" &&
    typeof value.title === "string" &&
    (value.url === null || typeof value.url === "string") &&
    typeof value.chunk_id === "string"
  );
}

function safeSourceUrl(value: string | null): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

function renderInlineMarkdown(text: string, sources: SourceCitation[]): ReactNode[] {
  return text.split(/(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*|\[Source \d+\])/g).map((part, index) => {
    const citationMatch = part.match(/^\[Source (\d+)\]$/);
    if (citationMatch) {
      const source = sources.find((item) => item.number === Number(citationMatch[1]));
      const href = safeSourceUrl(source?.url ?? null);
      return href ? (
        <a
          key={index}
          className="msg__citation-inline"
          href={href}
          target="_blank"
          rel="noopener noreferrer"
        >
          {part}
        </a>
      ) : (
        <span key={index}>{part}</span>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`")) {
      return <code key={index}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }
    return part;
  });
}

function renderMessageText(text: string, sources: SourceCitation[] = []): ReactNode[] {
  return text.split("\n").map((line, index, lines) => {
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const ordered = line.match(/^\s*(\d+[.)])\s+(.*)$/);
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const marker = bullet ? "•" : ordered?.[1];
    const content = bullet ? bullet[1] : ordered ? ordered[2] : heading ? heading[1] : line;

    return (
      <Fragment key={index}>
        {marker && <span className="msg__list-marker">{marker} </span>}
        {heading ? <strong>{renderInlineMarkdown(content, sources)}</strong> : renderInlineMarkdown(content, sources)}
        {index < lines.length - 1 ? "\n" : null}
      </Fragment>
    );
  });
}

type View = "home" | "chat";

const HISTORY: { id: number; title: string; date: string }[] = [
  { id: 1, title: "Campus library hours", date: "Today" },
  { id: 2, title: "Course registration help", date: "Yesterday" },
  { id: 3, title: "Hostel fee deadline", date: "Nov 10" },
];

export function ChatHome() {
  const [view, setView] = useState<View>("home");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [typing, setTyping] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const shouldFollowRef = useRef(true);
  const autoScrollRef = useRef(false);
  const sendingRef = useRef(false);

  useLayoutEffect(() => {
    const container = chatScrollRef.current;
    if (!container || !shouldFollowRef.current) return;

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const target = Math.max(0, container.scrollHeight - container.clientHeight);
    autoScrollRef.current = !reducedMotion && target > container.scrollTop + 1;
    container.scrollTo({ top: target, behavior: reducedMotion ? "auto" : "smooth" });
  }, [messages, typing]);

  function updateScrollPosition() {
    const container = chatScrollRef.current;
    if (!container) return;

    const distanceFromBottom = container.scrollHeight - container.clientHeight - container.scrollTop;
    if (autoScrollRef.current) {
      if (distanceFromBottom <= 2) autoScrollRef.current = false;
      return;
    }
    shouldFollowRef.current = distanceFromBottom < 96;
  }

  function interruptAutoScroll() {
    autoScrollRef.current = false;
  }

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  async function send() {
    const text = input.trim();
    if (!text || sendingRef.current) return;

    sendingRef.current = true;
    shouldFollowRef.current = true;
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    if (view === "home") setView("chat");

    const userMsg: Message = { id: uid(), role: "user", text, ts: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setTyping(true);

    try {
      const response = await fetch("/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          history: messages
            .filter((message) => !message.error)
            .slice(-8)
            .map(({ role, text }) => ({ role, content: text.slice(0, 2000) })),
        }),
      });
      const payload: unknown = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = isRecord(payload) && typeof payload.detail === "string"
          ? payload.detail
          : `JKLU Compass returned an error (${response.status}).`;
        throw new Error(detail);
      }

      if (!isRecord(payload) || typeof payload.answer !== "string") {
        throw new Error("JKLU Compass returned an invalid response. Please try again.");
      }

      const answer = payload.answer;
      const sources = Array.isArray(payload.sources)
        ? payload.sources.filter(isSourceCitation)
        : [];

      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "assistant", text: answer, ts: new Date(), sources },
      ]);
    } catch (error) {
      const errorText = error instanceof TypeError
        ? "Couldn't connect to JKLU Compass. Start the backend and confirm Ollama is running, then try again."
        : error instanceof Error
          ? error.message
          : "Something went wrong while contacting JKLU Compass.";
      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "assistant", text: errorText, ts: new Date(), error: true },
      ]);
    } finally {
      sendingRef.current = false;
      setTyping(false);
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  function fmt(d: Date) {
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  return (
    <>
      {/* ── Top bar ── */}
      <div className="topbar" style={{ pointerEvents: "all" }}>
        <div className="topbar__center">
          <button
            className={`history-btn ${historyOpen ? "history-btn--open" : ""}`}
            onClick={() => setHistoryOpen((o) => !o)}
            aria-expanded={historyOpen}
            aria-label="Chat history"
          >
            <HistoryIcon />
            <span>History</span>
            <ChevronIcon open={historyOpen} />
          </button>

          {historyOpen && (
            <div className="history-dropdown" role="menu">
              {HISTORY.map((h) => (
                <button
                  key={h.id}
                  className="history-item"
                  role="menuitem"
                  onClick={() => {
                    setHistoryOpen(false);
                    setView("chat");
                  }}
                >
                  <span className="history-item__title">{h.title}</span>
                  <span className="history-item__date">{h.date}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Home hero (shown before first message) ── */}
      {view === "home" && (
        <div className="hero" style={{ pointerEvents: "none" }}>
          <h1 className="hero__headline">what's on your mind?</h1>
        </div>
      )}

      {/* ── Chat messages (shown after first send) ── */}
      {view === "chat" && (
        <div
          ref={chatScrollRef}
          className="chat-scroll"
          style={{ pointerEvents: "all" }}
          onScroll={updateScrollPosition}
          onWheel={interruptAutoScroll}
          onTouchMove={interruptAutoScroll}
          onPointerDown={interruptAutoScroll}
        >
          <div className="chat-messages">
            {messages.map((m) => (
              <div key={m.id} className={`msg msg--${m.role}`}>
                {m.role === "assistant" && (
                  <div className="msg__avatar">◈</div>
                )}
                <div className="msg__bubble">
                  <p className={`msg__text${m.error ? " msg__text--error" : ""}`}>
                    {renderMessageText(m.text, m.sources)}
                  </p>
                  {m.sources && m.sources.length > 0 && (
                    <div className="msg__sources" role="group" aria-label="Sources">
                      {m.sources.map((source) => {
                        const href = safeSourceUrl(source.url);
                        const title = source.title.trim() || `Source ${source.number}`;
                        const label = `[Source ${source.number}] ${title}`;

                        return href ? (
                          <a
                            key={source.chunk_id}
                            className="msg__source"
                            href={href}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {label}
                          </a>
                        ) : (
                          <span key={source.chunk_id} className="msg__source msg__source--unlinked">
                            {label}
                          </span>
                        );
                      })}
                    </div>
                  )}
                  <span className="msg__ts">{fmt(m.ts)}</span>
                </div>
              </div>
            ))}
            {typing && (
              <div className="msg msg--assistant">
                <div className="msg__avatar">◈</div>
                <div className="msg__bubble" role="status" aria-label="JKLU Compass is responding">
                  <span className="typing-dots">
                    <span /><span /><span />
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Input bar (always at bottom-center) ── */}
      <div className="input-wrap" style={{ pointerEvents: "all" }}>
        <div className="input-box">
          <textarea
            ref={textareaRef}
            className="input-box__textarea"
            placeholder="Ask JKLU Compass anything..."
            rows={1}
            maxLength={2000}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              autoResize();
            }}
            onKeyDown={onKeyDown}
            aria-label="Message input"
          />
          <div className="input-box__toolbar">
            <div className="input-box__toolbar-left">
              <button className="icon-btn" aria-label="Attach">
                <PlusIcon />
              </button>
              <button className="icon-btn" aria-label="Voice input">
                <MicIcon />
              </button>
            </div>
            <div className="input-box__toolbar-right">
              <span className="input-hint">↵ send</span>
              <span className="input-hint-sep">·</span>
              <span className="input-hint">⇧↵ newline</span>
              <button
                className="send-btn"
                onClick={send}
                disabled={!input.trim() || typing}
                aria-label="Send message"
              >
                <SendIcon />
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

/* ── Icons ───────────────────────────────────────────────────── */

function HistoryIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M12 7v5l4 2" />
    </svg>
  );
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="12"
      height="12"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform 0.15s" }}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="9" y="2" width="6" height="11" rx="3" />
      <path d="M5 10a7 7 0 0 0 14 0" />
      <line x1="12" y1="19" x2="12" y2="22" />
      <line x1="8" y1="22" x2="16" y2="22" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 17 17 7" />
      <path d="M7 7h10v10" />
    </svg>
  );
}
