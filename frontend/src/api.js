const BASE = import.meta.env.VITE_API_URL || "";

async function j(res) {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return res.json();
}

function form(obj) {
  const f = new FormData();
  Object.entries(obj).forEach(([k, v]) => v !== undefined && v !== null && f.append(k, v));
  return f;
}

export const api = {
  health: () => fetch(`${BASE}/api/health`).then(j),
  models: () => fetch(`${BASE}/api/models`).then(j),
  benchmarks: () => fetch(`${BASE}/api/benchmarks`).then(j),
  jobs: () => fetch(`${BASE}/api/jobs`).then(j),
  job: (id) => fetch(`${BASE}/api/jobs/${id}`).then(j),
  frameStats: (id) => fetch(`${BASE}/api/jobs/${id}/frame_stats`).then(j),
  removeJob: (id) => fetch(`${BASE}/api/jobs/${id}`, { method: "DELETE" }).then(j),
  createJob: (file, o) => fetch(`${BASE}/api/jobs`, { method: "POST", body: form({ file, model: o.model, conf: o.conf, frame_stride: o.stride }) }).then(j),
  createDemoJob: (o) => fetch(`${BASE}/api/jobs/demo`, { method: "POST", body: form({ model: o.model, conf: o.conf, frame_stride: o.stride }) }).then(j),
  predictImage: (file, o) => fetch(`${BASE}/api/predict/image`, { method: "POST", body: form({ file, model: o.model, conf: o.conf, annotate: "false" }) }).then(j),
  sampleTruth: (seed) => fetch(`${BASE}/api/sample/${seed}/truth`).then(j),
  sampleBlob: (seed) =>
    fetch(`${BASE}/api/sample/${seed}/image`).then((r) => {
      if (!r.ok) throw new Error("could not load sample");
      return r.blob();
    }),
};

export const urls = {
  video: (id) => `${BASE}/api/jobs/${id}/video`,
  report: (id, ext) => `${BASE}/api/jobs/${id}/report.${ext}`,
  snapshot: (id, name) => `${BASE}/api/jobs/${id}/snapshots/${name}`,
  stream: (model, conf) => `${BASE}/api/stream/demo?model=${encodeURIComponent(model)}&conf=${conf}`,
};

export function wsUrl(path) {
  if (BASE) return BASE.replace(/^http/, "ws") + path;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}${path}`;
}
