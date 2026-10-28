export const CLASSES = ["scratch", "crack", "stain", "missing_part"];

export const CLASS_LABEL = {
  scratch: "Scratch",
  crack: "Crack",
  stain: "Stain",
  missing_part: "Missing part",
};

export const CLASS_COLOR = {
  scratch: "var(--c-scratch)",
  crack: "var(--c-crack)",
  stain: "var(--c-stain)",
  missing_part: "var(--c-missing)",
};

// hex twins for canvas drawing (canvas cannot read CSS variables in a loop cheaply)
export const CLASS_HEX = {
  scratch: "#f5b400",
  crack: "#e5484d",
  stain: "#2f81f7",
  missing_part: "#c13ed0",
};

export const fmt = {
  ms: (v) => (v == null ? "n/a" : `${v >= 100 ? v.toFixed(0) : v.toFixed(1)} ms`),
  pct: (v, d = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`),
  n3: (v) => (v == null ? "n/a" : v.toFixed(3)),
  time: (s) => {
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    return `${m}:${r.toFixed(1).padStart(4, "0")}`;
  },
};

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}
