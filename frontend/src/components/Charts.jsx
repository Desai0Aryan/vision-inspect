const nice = (max) => {
  const p = 10 ** Math.floor(Math.log10(max || 1));
  const n = max / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
};

export function LatencyLine({ stats, p50, p95 }) {
  const pts = stats.filter((s) => s.latency_ms != null);
  if (pts.length < 2) return <div className="empty">No latency samples.</div>;
  const W = 640;
  const H = 190;
  const m = { l: 44, r: 10, t: 12, b: 26 };
  const maxY = nice(Math.max(...pts.map((p) => p.latency_ms), p95) * 1.1);
  const lastF = pts[pts.length - 1].frame || 1;
  const X = (f) => m.l + (f / lastF) * (W - m.l - m.r);
  const Y = (v) => H - m.b - (v / maxY) * (H - m.t - m.b);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p.frame).toFixed(1)} ${Y(p.latency_ms).toFixed(1)}`).join("");
  const yt = [0, maxY / 2, maxY];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Inference latency per frame">
      {yt.map((v) => (
        <g key={v}>
          <line x1={m.l} x2={W - m.r} y1={Y(v)} y2={Y(v)} stroke="var(--line)" strokeDasharray={v ? "3 4" : ""} />
          <text x={m.l - 6} y={Y(v) + 4} textAnchor="end">{v.toFixed(0)}</text>
        </g>
      ))}
      <line x1={m.l} x2={W - m.r} y1={Y(p95)} y2={Y(p95)} stroke="var(--c-crack)" strokeDasharray="5 4" />
      <text x={W - m.r} y={Y(p95) - 5} textAnchor="end" style={{ fill: "var(--c-crack)" }}>{`p95 ${p95} ms`}</text>
      <line x1={m.l} x2={W - m.r} y1={Y(p50)} y2={Y(p50)} stroke="var(--ok)" strokeDasharray="5 4" />
      <text x={m.l + 4} y={Y(p50) + 15} style={{ fill: "var(--ok)" }}>{`median ${p50} ms`}</text>
      <path d={d} fill="none" stroke="var(--green-hi)" strokeWidth="1.6" strokeLinejoin="round" />
      <text x={m.l} y={H - 6}>frame 0</text>
      <text x={W - m.r} y={H - 6} textAnchor="end">{`frame ${lastF}  ·  latency in milliseconds`}</text>
      <text x={W - m.r} y={H - 6} textAnchor="end" style={{ display: "none" }} />
    </svg>
  );
}

/** Grouped vertical bars. series: [{name,color,values:[..]}], cats: [labels]. Values assumed 0..1 unless max given. */
export function GroupedBars({ cats, series, max = 1, fmtv = (v) => v.toFixed(2), height = 220 }) {
  const W = 560;
  const m = { l: 30, r: 8, t: 14, b: 28 };
  const gw = (W - m.l - m.r) / cats.length;
  const bw = Math.min(34, (gw - 14) / series.length);
  const Y = (v) => height - m.b - (v / max) * (height - m.t - m.b);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${height}`} role="img">
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line x1={m.l} x2={W - m.r} y1={Y(f * max)} y2={Y(f * max)} stroke="var(--line)" strokeDasharray={f ? "3 4" : ""} />
          <text x={m.l - 6} y={Y(f * max) + 4} textAnchor="end">{(f * max).toFixed(max > 1 ? 0 : 1)}</text>
        </g>
      ))}
      {cats.map((c, ci) => {
        const x0 = m.l + ci * gw + (gw - bw * series.length) / 2;
        return (
          <g key={c}>
            {series.map((s, si) => {
              const v = s.values[ci];
              if (v == null) return null;
              return (
                <g key={s.name}>
                  <rect x={x0 + si * bw} y={Y(v)} width={bw - 3} height={Math.max(0, Y(0) - Y(v))} rx="3" fill={s.color} />
                  <text x={x0 + si * bw + (bw - 3) / 2} y={Y(v) - 4} textAnchor="middle" style={{ fill: "var(--ink-2)", fontSize: 10.5 }}>{fmtv(v)}</text>
                </g>
              );
            })}
            <text x={m.l + ci * gw + gw / 2} y={height - 8} textAnchor="middle">{c}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((i) => (
        <span key={i.name}>
          <i className="swatch" style={{ background: i.color }} />
          {i.name}
        </span>
      ))}
    </div>
  );
}

/** Horizontal bars for latency: rows [{label, value, color, note}] */
export function HLatency({ rows }) {
  const max = Math.max(...rows.map((r) => r.value));
  return (
    <div>
      {rows.map((r) => (
        <div key={r.label} style={{ margin: "14px 0" }}>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 5 }}>
            <b>{r.label}</b>
            <span className="num">
              {r.value.toFixed(1)} ms <span className="muted small">{r.note}</span>
            </span>
          </div>
          <div className="hbar-track" style={{ height: 14 }}>
            <div className="hbar-fill" style={{ width: `${(r.value / max) * 100}%`, background: r.color }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export function Sparkline({ values, height = 44, color = "var(--green-hi)" }) {
  if (values.length < 2) return <div style={{ height }} />;
  const W = 240;
  const max = Math.max(...values) * 1.1;
  const d = values.map((v, i) => `${i ? "L" : "M"}${((i / (values.length - 1)) * W).toFixed(1)} ${(height - (v / max) * height).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} preserveAspectRatio="none" role="img" aria-label="Recent latency">
      <path d={d} fill="none" stroke={color} strokeWidth="1.8" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
