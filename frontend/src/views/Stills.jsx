import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { Icon, Seg, Slider } from "../components/ui.jsx";
import { CLASS_COLOR, CLASS_HEX, CLASS_LABEL, fmt, iou } from "../format.js";

export default function Stills({ models, defaultModel }) {
  const [model, setModel] = useState(defaultModel);
  const [conf, setConf] = useState(0.45);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 9000) + 100);
  const [img, setImg] = useState(null); // HTMLImageElement
  const [blob, setBlob] = useState(null);
  const [truth, setTruth] = useState(null);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [hot, setHot] = useState(false);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);

  const setBlobAsImage = (b) =>
    new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("That file is not an image the browser can read."));
      el.src = URL.createObjectURL(b);
    });

  const loadSample = useCallback(async (sd) => {
    setErr("");
    try {
      const [b, tr] = await Promise.all([api.sampleBlob(sd), api.sampleTruth(sd)]);
      setBlob(b);
      setTruth(tr.defects);
      setImg(await setBlobAsImage(b));
    } catch (e) {
      setErr(e.message);
    }
  }, []);

  useEffect(() => {
    loadSample(seed);
  }, [seed, loadSample]);

  const loadFile = async (f) => {
    setErr("");
    try {
      setBlob(f);
      setTruth(null);
      setImg(await setBlobAsImage(f));
    } catch (e) {
      setErr(e.message);
    }
  };

  useEffect(() => {
    if (!blob) return;
    let cancel = false;
    setBusy(true);
    api
      .predictImage(new File([blob], "part.jpg", { type: "image/jpeg" }), { model, conf })
      .then((r) => !cancel && (setRes(r), setErr("")))
      .catch((e) => !cancel && setErr(e.message))
      .finally(() => !cancel && setBusy(false));
    return () => {
      cancel = true;
    };
  }, [blob, model, conf]);

  // draw
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !img) return;
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const g = c.getContext("2d");
    g.drawImage(img, 0, 0);
    const lw = Math.max(2, c.width / 320);
    if (truth) {
      g.save();
      g.setLineDash([lw * 3, lw * 2]);
      g.strokeStyle = "rgba(255,255,255,0.9)";
      g.lineWidth = lw * 0.8;
      truth.forEach((t) => g.strokeRect(t.box[0], t.box[1], t.box[2] - t.box[0], t.box[3] - t.box[1]));
      g.restore();
    }
    (res?.detections || []).forEach((d) => {
      const col = CLASS_HEX[d.class];
      g.strokeStyle = col;
      g.lineWidth = lw;
      g.strokeRect(d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]);
      const label = `${CLASS_LABEL[d.class]} ${d.confidence.toFixed(2)}`;
      g.font = `600 ${Math.round(c.width / 46)}px sans-serif`;
      const w = g.measureText(label).width + 10;
      const h = Math.round(c.width / 38);
      const y = Math.max(h, d.box[1]);
      g.fillStyle = col;
      g.fillRect(d.box[0], y - h, w, h);
      g.fillStyle = "#101010";
      g.fillText(label, d.box[0] + 5, y - h * 0.28);
    });
  }, [img, truth, res]);

  let verdict = null;
  if (res && truth) {
    const dets = res.detections;
    const matched = new Set();
    let found = 0;
    truth.forEach((t) => {
      const k = dets.findIndex((d, i) => !matched.has(i) && d.class === t.class && iou(d.box, t.box) >= 0.5);
      if (k >= 0) {
        matched.add(k);
        found++;
      }
    });
    verdict = { found, total: truth.length, extra: dets.length - matched.size };
  }

  return (
    <div>
      <h1 className="title">Try it on a single part</h1>
      <p className="lede">Generate a fresh part and watch the model look for defects, or drop in your own image. White dashes show where the defects really are.</p>
      <div className="split">
        <div>
          <div
            className="stage"
            style={{ aspectRatio: "1 / 1", maxHeight: 640, margin: "0 auto" }}
            onDragOver={(e) => { e.preventDefault(); setHot(true); }}
            onDragLeave={() => setHot(false)}
            onDrop={(e) => { e.preventDefault(); setHot(false); e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]); }}
          >
            <canvas ref={canvasRef} aria-label="Part image with detections drawn on it" />
            {hot && <div className="drop drop-hot"><strong>Drop the image to inspect it</strong></div>}
          </div>
          {err && <p className="err" role="alert">{err}</p>}
        </div>
        <aside className="rail">
          <div className="panel stack" style={{ gap: 18 }}>
            <div className="stack">
              <button type="button" className="btn btn-wide" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>
                <Icon name="shuffle" /> Generate a new part
              </button>
              <button type="button" className="btn btn-quiet btn-wide" onClick={() => fileRef.current?.click()}>
                <Icon name="upload" /> Use my own image
              </button>
              <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files[0] && loadFile(e.target.files[0])} />
            </div>
            <div>
              <div className="field-label">Model</div>
              <Seg label="Model" value={model} onChange={setModel} options={models.map((m) => ({ value: m.id, label: { int8: "INT8", fp32: "FP32", pytorch: "PyTorch", classical: "Classical" }[m.id] || m.id }))} />
            </div>
            <Slider label="Confidence threshold" value={conf} onChange={setConf} min={0.1} max={0.9} step={0.05} format={(v) => v.toFixed(2)} />
          </div>

          <div className="panel" aria-live="polite">
            <h2 className="h2">{busy && !res ? "Inspecting…" : res ? (res.defective ? "Defects detected" : "No defects detected") : "Result"}</h2>
            {res && (
              <>
                <p className="muted small" style={{ marginTop: 0 }}>{fmt.ms(res.latency_ms)} on {res.model.toUpperCase()}</p>
                {verdict && (
                  <p style={{ margin: "0 0 12px", fontWeight: 600 }}>
                    Found {verdict.found} of {verdict.total} real defect{verdict.total === 1 ? "" : "s"}
                    {verdict.extra > 0 ? `, plus ${verdict.extra} false alarm${verdict.extra === 1 ? "" : "s"}` : verdict.total === 0 ? ", and raised no false alarms" : ", with no false alarms"}.
                  </p>
                )}
                {res.detections.map((d, i) => (
                  <div className="hbar" key={i} style={{ gridTemplateColumns: "100px 1fr 40px" }}>
                    <span className="tag"><i className="swatch" style={{ background: CLASS_COLOR[d.class] }} />{CLASS_LABEL[d.class]}</span>
                    <div className="hbar-track"><div className="hbar-fill" style={{ width: `${d.confidence * 100}%`, background: CLASS_COLOR[d.class] }} /></div>
                    <span className="num">{d.confidence.toFixed(2)}</span>
                  </div>
                ))}
              </>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
