import { CLASS_COLOR, CLASS_LABEL } from "../format.js";

/**
 * The conveyor, drawn as a timeline. Every part that crossed the camera is a plate
 * sitting at the moment it passed the middle of the frame; defects are marks on the belt.
 * Click anywhere to seek the video.
 */
export default function BeltStrip({ parts, tracks, fps, duration, current, onSeek }) {
  const W = 1000;
  const H = 128;
  const pad = 36;
  const x = (t) => pad + (Math.min(Math.max(t, 0), duration) / duration) * (W - pad * 2);
  const beltY = 78;

  const seek = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    onSeek(Math.min(Math.max(((px - pad) / (W - pad * 2)) * duration, 0), duration));
  };

  const ticks = [];
  const step = duration > 40 ? 10 : duration > 16 ? 4 : 2;
  for (let t = 0; t <= duration; t += step) ticks.push(t);

  return (
    <div className="belt">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Conveyor timeline: each plate is one inspected part" onClick={seek}>
        <rect x={pad - 12} y={beltY - 6} width={W - pad * 2 + 24} height="12" rx="6" fill="var(--sunken)" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={beltY + 8} y2={beltY + 14} stroke="var(--line)" strokeWidth="2" />
            <text x={x(t)} y={beltY + 28} textAnchor="middle">{`${t}s`}</text>
          </g>
        ))}

        {parts.map((p) => {
          const mid = (p.first_frame + p.last_frame) / 2 / fps;
          const bad = p.verdict === "NOK";
          const cx = x(mid);
          return (
            <g key={p.id} transform={`translate(${cx - 30} ${beltY - 62})`} onClick={(e) => { e.stopPropagation(); onSeek(mid); }} style={{ cursor: "pointer" }}>
              <title>{`Part ${p.id}: ${bad ? `rejected, ${p.defects} defect${p.defects === 1 ? "" : "s"}` : "passed"}`}</title>
              <rect width="60" height="50" rx="5" fill="var(--green)" />
              <circle cx="8" cy="8" r="3" fill="var(--on-green)" opacity="0.7" />
              <circle cx="52" cy="8" r="3" fill="var(--on-green)" opacity="0.7" />
              <circle cx="8" cy="42" r="3" fill="var(--on-green)" opacity="0.7" />
              <circle cx="52" cy="42" r="3" fill="var(--on-green)" opacity="0.7" />
              <rect x="20" y="16" width="20" height="18" rx="2" fill="#10261a" opacity="0.85" />
              {bad ? (
                <g transform="rotate(-9 30 25)">
                  <rect x="4" y="15" width="52" height="20" rx="3" fill="var(--sheet)" stroke="var(--nok)" strokeWidth="2.5" />
                  <text x="30" y="30" textAnchor="middle" style={{ fill: "var(--nok)", fontWeight: 800, fontSize: 12, letterSpacing: "0.04em" }}>REJECT</text>
                </g>
              ) : (
                <g transform="rotate(-9 30 25)">
                  <rect x="12" y="15" width="36" height="20" rx="3" fill="var(--sheet)" stroke="var(--ok)" strokeWidth="2.5" />
                  <text x="30" y="30" textAnchor="middle" style={{ fill: "var(--ok)", fontWeight: 800, fontSize: 12, letterSpacing: "0.04em" }}>OK</text>
                </g>
              )}
              <text x="30" y="-5" textAnchor="middle">{`Part ${p.id}`}</text>
            </g>
          );
        })}

        {tracks.map((t) => (
          <g key={t.id}>
            <title>{`${CLASS_LABEL[t.class]} at ${t.first_time_s}s (confidence ${t.max_conf.toFixed(2)})`}</title>
            <path d={`M${x(t.first_time_s)} ${beltY - 9} l6 9 l-6 9 l-6 -9 z`} fill={CLASS_COLOR[t.class]} stroke="var(--sheet)" strokeWidth="1.5" />
          </g>
        ))}

        <g style={{ pointerEvents: "none" }}>
          <line x1={x(current)} x2={x(current)} y1={4} y2={beltY + 16} stroke="var(--ink)" strokeWidth="2" />
          <circle cx={x(current)} cy={4} r="4" fill="var(--ink)" />
        </g>
      </svg>
      <div className="belt-legend">
        {Object.keys(CLASS_LABEL).map((c) => (
          <span key={c}>
            <i className="swatch" style={{ background: CLASS_COLOR[c] }} />
            {CLASS_LABEL[c]}
          </span>
        ))}
      </div>
    </div>
  );
}
