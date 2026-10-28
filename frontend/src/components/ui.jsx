import { CLASS_COLOR, CLASS_LABEL, CLASSES } from "../format.js";

export function Icon({ name, size = 16 }) {
  const p = {
    play: "M7 4.5v15l12-7.5z",
    upload: "M12 16V4m0 0L7 9m5-5 5 5M5 20h14",
    download: "M12 4v12m0 0 5-5m-5 5-5-5M5 20h14",
    camera: "M4 8h3l1.5-2h7L17 8h3v11H4zM12 17a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z",
    stop: "M7 7h10v10H7z",
    alert: "M12 4 3 20h18zM12 10v5m0 3v.01",
    shuffle: "M4 7h4l8 10h4M4 17h4l2-2.5M14 9.5 16 7h4M18 5l2 2-2 2M18 15l2 2-2 2",
  }[name];
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={p} fill={name === "play" || name === "stop" ? "currentColor" : "none"} />
    </svg>
  );
}

export function Seg({ value, onChange, options, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" className="seg-btn" aria-pressed={value === o.value} onClick={() => onChange(o.value)} title={o.title}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Slider({ label, value, onChange, min, max, step, format }) {
  const id = `sl-${label.replace(/\W/g, "")}`;
  return (
    <div>
      <label className="field-label" htmlFor={id}>
        <span>{label}</span>
        <span className="num">{format ? format(value) : value}</span>
      </label>
      <input id={id} className="range" type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Ledger({ items }) {
  return (
    <div className="ledger">
      {items.map((it) => (
        <div className="ledger-item" key={it.label}>
          <div className={`ledger-value ${it.tone ? `tone-${it.tone}` : ""}`}>{it.value}</div>
          <div className="ledger-label">{it.label}</div>
        </div>
      ))}
    </div>
  );
}

export function ClassBars({ byClass }) {
  const max = Math.max(1, ...CLASSES.map((c) => byClass[c] || 0));
  return (
    <div>
      {CLASSES.map((c) => (
        <div className="hbar" key={c}>
          <span>{CLASS_LABEL[c]}</span>
          <div className="hbar-track">
            <div className="hbar-fill" style={{ width: `${((byClass[c] || 0) / max) * 100}%`, background: CLASS_COLOR[c] }} />
          </div>
          <span className="num">{byClass[c] || 0}</span>
        </div>
      ))}
    </div>
  );
}

export function Verdict({ v }) {
  return <span className={`verdict ${v === "OK" ? "verdict-ok" : "verdict-nok"}`}>{v === "OK" ? "OK" : "Rejected"}</span>;
}
