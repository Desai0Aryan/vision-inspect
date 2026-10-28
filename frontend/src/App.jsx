import { useEffect, useState } from "react";
import { api } from "./api.js";
import Inspect from "./views/Inspect.jsx";
import Live from "./views/Live.jsx";
import Models from "./views/Models.jsx";
import Stills from "./views/Stills.jsx";

const TABS = [
  { id: "inspect", label: "Inspect a clip" },
  { id: "live", label: "Live" },
  { id: "stills", label: "Single part" },
  { id: "models", label: "Models" },
];

const fromHash = () => (TABS.find((t) => `#${t.id}` === location.hash)?.id) || "inspect";

function Mark() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="var(--green)" />
      <circle cx="16" cy="16" r="6" fill="none" stroke="var(--on-green)" strokeWidth="2.5" />
      <path d="M16 4v5M16 23v5M4 16h5M23 16h5" stroke="var(--on-green)" strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState(fromHash);
  const [health, setHealth] = useState(null);
  const [models, setModels] = useState([]);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const on = () => setTab(fromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    let alive = true;
    const ping = () =>
      Promise.all([api.health(), api.models()])
        .then(([h, m]) => alive && (setHealth(h), setModels(m.models), setDown(false)))
        .catch(() => alive && setDown(true));
    ping();
    const iv = setInterval(ping, 10000);
    return () => {
      alive = false;
      clearInterval(iv);
    };
  }, []);

  const go = (id) => {
    location.hash = id;
    setTab(id);
  };

  const ready = health && models.length > 0;

  return (
    <div className="shell">
      <header className="bar">
        <div className="brand">
          <Mark /> VisionInspect
        </div>
        <nav className="tabs" role="tablist" aria-label="Sections">
          {TABS.map((t) => (
            <button key={t.id} role="tab" type="button" className="tab" aria-selected={tab === t.id} onClick={() => go(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
        <div className="pill" title="Server status">
          <span className={`dot ${down ? "dot-off" : health ? "dot-on" : ""}`} />
          {down ? "Server offline" : health ? `Server ready · default ${health.default_model.toUpperCase()}` : "Connecting…"}
        </div>
      </header>

      {down && (
        <div className="offline" role="alert">
          The dashboard cannot reach the API. Start it with <code>make serve</code> (or <code>uvicorn api.main:app --port 8000</code>), then this page will reconnect by itself.
        </div>
      )}

      <main>
        {ready && tab === "inspect" && <Inspect models={models} defaultModel={health.default_model} />}
        {ready && tab === "live" && <Live models={models} defaultModel={health.default_model} />}
        {ready && tab === "stills" && <Stills models={models} defaultModel={health.default_model} />}
        {tab === "models" && <Models />}
      </main>
    </div>
  );
}
