import { useEffect, useRef, useState } from "react";
import { urls, wsUrl } from "../api.js";
import { Sparkline } from "../components/Charts.jsx";
import { Icon, Ledger, Seg, Slider } from "../components/ui.jsx";

export default function Live({ models, defaultModel }) {
  const [mode, setMode] = useState("line");
  const [model, setModel] = useState(defaultModel);
  const [conf, setConf] = useState(0.45);
  const [applied, setApplied] = useState(0.45);

  useEffect(() => {
    const t = setTimeout(() => setApplied(conf), 450);
    return () => clearTimeout(t);
  }, [conf]);

  return (
    <div>
      <h1 className="title">Watch the line live</h1>
      <p className="lede">The demo line streams straight from the server, inspected as it plays. Switch to camera to send your own webcam frames and see round-trip latency.</p>
      <div className="split">
        <div>
          {mode === "line" ? (
            <div className="stage">
              <img key={`${model}-${applied}`} src={urls.stream(model, applied)} alt="Live inspection of the demo conveyor with detections drawn on each part" />
            </div>
          ) : (
            <Camera model={model} conf={applied} />
          )}
        </div>
        <aside className="rail">
          <div className="panel stack" style={{ gap: 18 }}>
            <div>
              <div className="field-label">Source</div>
              <Seg label="Source" value={mode} onChange={setMode} options={[{ value: "line", label: "Demo line" }, { value: "camera", label: "My camera" }]} />
            </div>
            <div>
              <div className="field-label">Model</div>
              <Seg label="Model" value={model} onChange={setModel} options={models.map((m) => ({ value: m.id, label: { int8: "INT8", fp32: "FP32", pytorch: "PyTorch", classical: "Classical" }[m.id] || m.id }))} />
            </div>
            <Slider label="Confidence threshold" value={conf} onChange={setConf} min={0.1} max={0.9} step={0.05} format={(v) => v.toFixed(2)} />
            <p className="muted small" style={{ margin: 0 }}>
              {mode === "line" ? "The speed, part and defect counts are drawn on the video itself." : "Frames are compressed to JPEG, sent over a WebSocket, inspected on the server and sent back with the boxes drawn in."}
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}

function Camera({ model, conf }) {
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const aliveRef = useRef(false);
  const [on, setOn] = useState(false);
  const [err, setErr] = useState("");
  const [frame, setFrame] = useState(null);
  const [info, setInfo] = useState(null);
  const [lat, setLat] = useState([]);
  const [rtt, setRtt] = useState(null);

  const stop = () => {
    aliveRef.current = false;
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setOn(false);
  };
  useEffect(() => stop, []);

  const start = async () => {
    setErr("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setErr("This browser cannot open a camera here. Camera access needs HTTPS or localhost.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    } catch (e) {
      setErr(`Could not open the camera: ${e.message}. Allow camera access in the address bar and try again.`);
      return;
    }
    const ws = new WebSocket(wsUrl(`/ws/live?model=${model}&conf=${conf}`));
    wsRef.current = ws;
    aliveRef.current = true;
    const canvas = document.createElement("canvas");
    let sentAt = 0;
    const send = () => {
      const v = videoRef.current;
      if (!aliveRef.current || !v || ws.readyState !== 1) return;
      canvas.width = 640;
      canvas.height = Math.round((640 * v.videoHeight) / v.videoWidth) || 480;
      canvas.getContext("2d").drawImage(v, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((b) => {
        if (!b || !aliveRef.current) return;
        sentAt = performance.now();
        ws.send(b);
      }, "image/jpeg", 0.72);
    };
    ws.onopen = () => {
      setOn(true);
      send();
    };
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.error) return send();
      setRtt(Math.round(performance.now() - sentAt));
      setFrame(`data:image/jpeg;base64,${m.annotated_b64}`);
      setInfo(m);
      setLat((l) => [...l.slice(-59), m.latency_ms]);
      setTimeout(send, 0);
    };
    ws.onclose = () => aliveRef.current && stop();
    ws.onerror = () => setErr("The live connection to the server failed.");
  };

  return (
    <div>
      <div className="stage">
        <video ref={videoRef} muted playsInline style={{ display: "none" }} />
        {frame ? (
          <img src={frame} alt="Your camera with detections drawn on it" />
        ) : (
          <div className="drop" onClick={start} role="button" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && start()}>
            <Icon name="camera" size={28} />
            <strong>Start your camera</strong>
            <span>The browser will ask for permission first.</span>
          </div>
        )}
      </div>
      <div className="row" style={{ marginTop: 14 }}>
        {on ? (
          <button type="button" className="btn btn-quiet" onClick={stop}><Icon name="stop" /> Stop camera</button>
        ) : (
          <button type="button" className="btn" onClick={start}><Icon name="camera" /> Start camera</button>
        )}
      </div>
      {err && <p className="err" role="alert">{err}</p>}
      {info && (
        <>
          <Ledger items={[
            { value: `${info.latency_ms}`, label: "ms model time" },
            { value: rtt ?? "n/a", label: "ms round trip" },
            { value: info.detections.length, label: "detections in view" },
            { value: info.rejected, label: "parts rejected" },
          ]} />
          <Sparkline values={lat} />
        </>
      )}
    </div>
  );
}
