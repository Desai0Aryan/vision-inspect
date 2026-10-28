import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, urls, wsUrl } from "../api.js";
import BeltStrip from "../components/BeltStrip.jsx";
import { LatencyLine } from "../components/Charts.jsx";
import { ClassBars, Icon, Ledger, Seg, Slider, Verdict } from "../components/ui.jsx";
import { CLASS_COLOR, CLASS_LABEL, fmt } from "../format.js";

const STRIDES = [
  { value: 1, label: "Every frame", title: "Highest accuracy, slowest" },
  { value: 2, label: "Every 2nd", title: "About twice as fast" },
  { value: 3, label: "Every 3rd", title: "Real time on a small CPU; the tracker fills the gaps" },
];

export default function Inspect({ models, defaultModel }) {
  const [model, setModel] = useState(defaultModel);
  const [conf, setConf] = useState(0.45);
  const [stride, setStride] = useState(2);
  const [hot, setHot] = useState(false);
  const [job, setJob] = useState(null); // full job object
  const [live, setLive] = useState(null); // {progress..., preview}
  const [stats, setStats] = useState([]);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [t, setT] = useState(0);
  const videoRef = useRef(null);
  const fileRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => setModel((m) => m || defaultModel), [defaultModel]);

  const refreshHistory = useCallback(() => api.jobs().then(setHistory).catch(() => {}), []);
  useEffect(() => {
    refreshHistory();
  }, [refreshHistory]);
  useEffect(() => () => wsRef.current?.close(), []);

  const load = useCallback(async (id) => {
    try {
      const j = await api.job(id);
      setJob(j);
      setLive(null);
      setError("");
      if (j.status === "done") api.frameStats(id).then(setStats).catch(() => setStats([]));
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const watch = useCallback(
    (id) => {
      wsRef.current?.close();
      const ws = new WebSocket(wsUrl(`/ws/jobs/${id}`));
      wsRef.current = ws;
      ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        setLive(m);
        if (m.status === "done" || m.status === "failed" || m.status === "cancelled") {
          load(id).then(refreshHistory);
        }
      };
      ws.onerror = () => {
        // fall back to polling if the socket is blocked (proxies, some corporate networks)
        const iv = setInterval(async () => {
          const j = await api.job(id).catch(() => null);
          if (!j) return;
          setLive({ status: j.status, ...(j.progress || {}) });
          if (["done", "failed", "cancelled"].includes(j.status)) {
            clearInterval(iv);
            load(id).then(refreshHistory);
          }
        }, 800);
      };
    },
    [load, refreshHistory],
  );

  const start = async (file) => {
    setError("");
    setJob(null);
    setStats([]);
    setT(0);
    try {
      const j = file ? await api.createJob(file, { model, conf, stride }) : await api.createDemoJob({ model, conf, stride });
      setJob(j);
      setLive({ status: "queued", frame: 0, total: 0 });
      watch(j.id);
      refreshHistory();
    } catch (e) {
      setError(e.message);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setHot(false);
    const f = e.dataTransfer.files?.[0];
    if (f) start(f);
  };

  const running = job && (job.status === "queued" || job.status === "running");
  const done = job?.status === "done";
  const s = done ? job.summary : null;

  const seek = (sec) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = sec;
    setT(sec);
  };

  const ledger = useMemo(() => {
    if (!s) return [];
    const p = s.parts;
    return [
      { value: p.inspected, label: "parts inspected" },
      { value: p.rejected, label: "rejected", tone: p.rejected ? "bad" : "good" },
      { value: p.yield_pct == null ? "n/a" : `${p.yield_pct}%`, label: "yield" },
      { value: s.defects.unique, label: "unique defects" },
      { value: `${s.performance.latency_ms_p50}`, label: "ms per frame (median)" },
      { value: `${s.performance.realtime_factor}x`, label: "real time", tone: s.performance.realtime_factor >= 1 ? "good" : undefined },
    ];
  }, [s]);

  const progressPct = live?.total ? Math.min(100, (live.frame / live.total) * 100) : 0;

  return (
    <div>
      <h1 className="title">Inspect a clip, part by part</h1>
      <p className="lede">Upload a video of an assembly line, or run the built-in conveyor clip. Every part gets a pass or reject verdict, and every defect is counted once.</p>

      <div className="split">
        <div>
          <div className="stage" onDragOver={(e) => { e.preventDefault(); setHot(true); }} onDragLeave={() => setHot(false)} onDrop={onDrop}>
            {done ? (
              <video ref={videoRef} src={urls.video(job.id)} controls onTimeUpdate={(e) => setT(e.currentTarget.currentTime)} onLoadedMetadata={(e) => { const first = s?.tracks?.[0]?.first_time_s; if (first > 1) e.currentTarget.currentTime = Math.max(0, first - 0.8); }} aria-label="Annotated inspection video" />
            ) : running && live?.preview_b64 ? (
              <img src={`data:image/jpeg;base64,${live.preview_b64}`} alt="Live preview of the clip being inspected" />
            ) : (
              <div className={`drop ${hot ? "drop-hot" : ""}`} onClick={() => !running && fileRef.current?.click()} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}>
                {running ? (
                  <strong>Starting the inspection…</strong>
                ) : (
                  <>
                    <Icon name="upload" size={28} />
                    <strong>Drop a video here</strong>
                    <span>or choose a file. MP4, MOV and WebM work.</span>
                  </>
                )}
              </div>
            )}
            {running && (
              <div className="progress" role="status">
                <div className="bar-track"><div className="bar-fill" style={{ width: `${progressPct}%` }} /></div>
                <span className="num">
                  {live?.status === "queued" ? "Waiting for the worker" : `Frame ${live?.frame ?? 0} of ${live?.total || "…"}`}
                  {live?.fps ? ` · ${live.fps} fps · ${live.latency_ms} ms per frame · ${live.parts} parts · ${live.rejected} rejected` : ""}
                </span>
              </div>
            )}
          </div>
          <input ref={fileRef} type="file" accept="video/*" hidden onChange={(e) => e.target.files[0] && start(e.target.files[0])} />
          {error && <p className="err" role="alert">{error}</p>}
          {job?.status === "failed" && <p className="err" role="alert">The inspection failed: {job.error}</p>}
        </div>

        <aside className="rail">
          <div className="panel stack" style={{ gap: 18 }}>
            <div>
              <div className="field-label">Model</div>
              <Seg label="Model" value={model} onChange={setModel} options={models.map((m) => ({ value: m.id, label: { int8: "INT8", fp32: "FP32", pytorch: "PyTorch", classical: "Classical" }[m.id] || m.id, title: `${m.label}, ${m.size_mb} MB` }))} />
              <div className="muted small" style={{ marginTop: 6 }}>{models.find((m) => m.id === model)?.label}</div>
            </div>
            <Slider label="Confidence threshold" value={conf} onChange={setConf} min={0.1} max={0.9} step={0.05} format={(v) => v.toFixed(2)} />
            <div>
              <div className="field-label">Run the detector</div>
              <Seg label="Frame stride" value={stride} onChange={setStride} options={STRIDES} />
              <div className="muted small" style={{ marginTop: 6 }}>{STRIDES.find((x) => x.value === stride)?.title}.</div>
            </div>
            <div className="stack">
              <button type="button" className="btn btn-wide" disabled={running} onClick={() => start(null)}>
                <Icon name="play" /> Run the demo conveyor
              </button>
              <button type="button" className="btn btn-quiet btn-wide" disabled={running} onClick={() => fileRef.current?.click()}>
                <Icon name="upload" /> Upload a video
              </button>
              {running && (
                <button type="button" className="btn btn-quiet btn-wide" onClick={() => api.removeJob(job.id)}>
                  Cancel
                </button>
              )}
            </div>
          </div>

          <div className="panel">
            <h2 className="h2">Recent runs</h2>
            {history.length === 0 ? (
              <p className="muted small" style={{ margin: 0 }}>Nothing yet. Runs you start appear here.</p>
            ) : (
              <div className="history">
                {history.slice(0, 6).map((h) => (
                  <button key={h.id} type="button" className="hist-item" aria-current={job?.id === h.id} onClick={() => load(h.id)}>
                    <span>{h.filename || h.id}</span>
                    <span className="muted">{h.status === "done" ? h.model : h.status}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </aside>
      </div>

      {s && (
        <section aria-label="Inspection results">
          {s.alert.triggered && (
            <div className="alert" role="alert">
              <Icon name="alert" size={18} />
              Defect rate alert: {s.parts.defect_rate_pct}% of parts were rejected, above the {s.alert.threshold_pct.toFixed(0)}% limit.
            </div>
          )}
          <Ledger items={ledger} />

          <h2 className="h2" style={{ marginTop: 28 }}>The line, in order</h2>
          <p className="muted small" style={{ margin: "0 0 4px" }}>Select a plate or a defect mark to jump the video to that moment.</p>
          <BeltStrip parts={s.parts_detail} tracks={s.tracks} fps={s.video.fps} duration={s.video.duration_s} current={t} onSeek={seek} />

          <div className="cols-2">
            <div>
              <h2 className="h2">Defects found</h2>
              {s.tracks.length === 0 ? (
                <div className="empty">No defects were confirmed in this clip.</div>
              ) : (
                <div className="gallery">
                  {s.tracks.filter((x) => x.snapshot).map((d) => (
                    <button key={d.id} type="button" className="shot" onClick={() => seek(Math.max(0, d.first_time_s - 0.2))}>
                      <img src={urls.snapshot(job.id, d.snapshot)} alt={`${CLASS_LABEL[d.class]} on part ${d.part_id ?? "unknown"}`} loading="lazy" />
                      <div className="shot-cap">
                        <span className="tag">
                          <i className="swatch" style={{ background: CLASS_COLOR[d.class] }} />
                          {CLASS_LABEL[d.class]}
                        </span>
                        <span className="num muted">{d.max_conf.toFixed(2)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div>
              <h2 className="h2">By type</h2>
              <ClassBars byClass={s.defects.by_class} />
              <h2 className="h2" style={{ marginTop: 26 }}>Parts</h2>
              <table className="table">
                <thead><tr><th>Part</th><th>Verdict</th><th>Defects</th></tr></thead>
                <tbody>
                  {s.parts_detail.map((p) => (
                    <tr key={p.id}>
                      <td>Part {p.id}</td>
                      <td><Verdict v={p.verdict} /></td>
                      <td>{p.defects}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="cols-2">
            <div>
              <h2 className="h2">Speed</h2>
              {stats.length ? <LatencyLine stats={stats} p50={s.performance.latency_ms_p50} p95={s.performance.latency_ms_p95} /> : <div className="empty">Latency details are not available for this run.</div>}
              <p className="muted small">
                {s.model.toUpperCase()} · detector ran on {s.performance.frames_inferred} of {s.video.frames} frames · {s.performance.end_to_end_fps} fps end to end, {fmt.ms(s.performance.latency_ms_p95)} at the 95th percentile.
              </p>
            </div>
            <div>
              <h2 className="h2">Take the results with you</h2>
              <div className="row" style={{ flexWrap: "wrap" }}>
                <a className="link-btn" href={urls.report(job.id, "pdf")} download><Icon name="download" /> PDF report</a>
                <a className="link-btn" href={urls.report(job.id, "csv")} download><Icon name="download" /> CSV log</a>
                <a className="link-btn" href={urls.report(job.id, "json")} download><Icon name="download" /> JSON</a>
              </div>
              <p className="muted small" style={{ maxWidth: "44ch" }}>The PDF has the summary, charts, part verdicts and the best crop of every defect. The CSV has one row per tracked detection.</p>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
