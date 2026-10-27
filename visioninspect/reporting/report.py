"""Downloadable inspection reports: JSON (machine), CSV (per-detection log), PDF (human, with charts + snapshots)."""
from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from visioninspect import CLASSES  # noqa: E402

CLS_HEX = {"scratch": "#f5b400", "crack": "#e5484d", "stain": "#2f81f7", "missing_part": "#c13ed0"}
INK, MUTED, ACCENT = colors.HexColor("#1b1f24"), colors.HexColor("#6b7280"), colors.HexColor("#2f81f7")


def write_json(out_dir: Path, summary: dict, events: list, frame_stats: list):
    (out_dir / "report.json").write_text(json.dumps({"summary": summary, "events": events}, indent=2))
    (out_dir / "frame_stats.json").write_text(json.dumps(frame_stats))


def write_csv(out_dir: Path, events: list):
    cols = ["frame", "time_s", "track_id", "class", "confidence", "x1", "y1", "x2", "y2"]
    with open(out_dir / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for e in events:
            w.writerow({k: e.get(k) for k in cols})


def _charts(out_dir: Path, summary: dict, frame_stats: list):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    # defects by class
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    by = summary["defects"]["by_class"]
    ax.barh(CLASSES, [by[c] for c in CLASSES], color=[CLS_HEX[c] for c in CLASSES])
    for i, c in enumerate(CLASSES):
        ax.text(by[c] + 0.05, i, str(by[c]), va="center")
    ax.invert_yaxis()
    ax.set_title("Unique defects by class")
    ax.set_xlim(0, max(max(by.values()), 1) * 1.2)
    fig.tight_layout()
    fig.savefig(out_dir / "chart_classes.png", dpi=170)
    plt.close(fig)
    # latency timeline
    lat = [(f["frame"], f["latency_ms"]) for f in frame_stats if f["latency_ms"] is not None]
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    if lat:
        xs, ys = zip(*lat)
        ax.plot(xs, ys, color="#2f81f7", lw=1)
        p = summary["performance"]
        ax.axhline(p["latency_ms_p95"], color="#e5484d", ls="--", lw=1, label=f"p95 {p['latency_ms_p95']} ms")
        ax.axhline(p["latency_ms_mean"], color="#2da44e", ls="--", lw=1, label=f"mean {p['latency_ms_mean']} ms")
        ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("frame")
    ax.set_ylabel("inference ms")
    ax.set_title("Inference latency over the clip")
    fig.tight_layout()
    fig.savefig(out_dir / "chart_latency.png", dpi=170)
    plt.close(fig)


def write_pdf(out_dir: Path, summary: dict, frame_stats: list, title: str = "Inspection report", source_name: str = ""):
    _charts(out_dir, summary, frame_stats)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=22, leading=28, textColor=INK, alignment=0, spaceAfter=6)
    sub = ParagraphStyle("sub", parent=ss["Normal"], fontSize=9, textColor=MUTED)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=INK, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12, textColor=INK)
    doc = SimpleDocTemplate(str(out_dir / "report.pdf"), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title=title)
    P, perf, v = summary["parts"], summary["performance"], summary["video"]
    story = [Paragraph(title, h1),
             Paragraph(f"{source_name or 'uploaded clip'} &nbsp;·&nbsp; model: <b>{summary['model']}</b> &nbsp;·&nbsp; "
                       f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;·&nbsp; {v['width']}x{v['height']} @ {v['fps']} fps, {v['duration_s']} s", sub),
             Spacer(1, 6)]
    if summary["alert"]["triggered"]:
        t = Table([[f"DEFECT RATE ALERT: {P['defect_rate_pct']}% of parts rejected (threshold {summary['alert']['threshold_pct']:.0f}%)"]], colWidths=[178 * mm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdecec")), ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#b42318")),
                               ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9), ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e5484d")),
                               ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story += [t, Spacer(1, 6)]

    def kpi(label, value):
        return [Paragraph(f'<font size="17"><b>{value}</b></font><br/><font size="8" color="#6b7280">{label}</font>', body)]
    kp = Table([[kpi("parts inspected", P["inspected"]), kpi("rejected", P["rejected"]), kpi("yield", f"{P['yield_pct']}%" if P["yield_pct"] is not None else "n/a"),
                 kpi("unique defects", summary["defects"]["unique"]), kpi("mean latency", f"{perf['latency_ms_mean']} ms"), kpi("end-to-end", f"{perf['end_to_end_fps']} fps")]],
               colWidths=[29.6 * mm] * 6)
    kp.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")), ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
                            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story += [kp, Paragraph("Summary", h2)]
    story.append(Paragraph(
        f"{P['inspected']} parts crossed the line; <b>{P['rejected']}</b> were rejected and {P['passed']} passed. "
        f"The tracker merged repeated sightings into <b>{summary['defects']['unique']}</b> unique defects. "
        f"Inference ran at {perf['inference_fps']} fps (p50 {perf['latency_ms_p50']} ms, p95 {perf['latency_ms_p95']} ms, max {perf['latency_ms_max']} ms); "
        f"the full pipeline processed the clip at {perf['realtime_factor']}x real time.", body))
    story += [Spacer(1, 6), Table([[Image(str(out_dir / "chart_classes.png"), 80 * mm, 45.7 * mm), Image(str(out_dir / "chart_latency.png"), 96 * mm, 32 * mm)]], colWidths=[84 * mm, 98 * mm])]

    story.append(Paragraph("Part verdicts", h2))
    rows = [["Part", "Verdict", "Defects", "Frames"]] + [[f"#{p['id']}", p["verdict"], p["defects"], f"{p['first_frame']}-{p['last_frame']}"] for p in summary["parts_detail"]]
    t = Table(rows, colWidths=[25 * mm, 30 * mm, 25 * mm, 40 * mm], hAlign="LEFT")
    st = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de"))]
    for i, p in enumerate(summary["parts_detail"], 1):
        st.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor("#b42318") if p["verdict"] == "NOK" else colors.HexColor("#1a7f37")))
    t.setStyle(TableStyle(st))
    story.append(t)

    if summary["tracks"]:
        story += [PageBreak(), Paragraph("Defect log", h2)]
        rows = [["ID", "Class", "Part", "Time (s)", "Mean conf", "Max conf", "Frames seen"]]
        rows += [[t["id"], t["class"], t["part_id"] or "-", t["first_time_s"], f"{t['mean_conf']:.2f}", f"{t['max_conf']:.2f}", t["hits"]] for t in summary["tracks"]]
        t = Table(rows, hAlign="LEFT")
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                               ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de"))]))
        story.append(t)
        snaps = [s for s in summary["snapshots"] if (out_dir / "snapshots" / s).exists()]
        if snaps:
            story.append(Paragraph("Best detection per defect", h2))
            cells, row = [], []
            for s in snaps:
                tid = int(s.split("_")[1].split(".")[0])
                tr = next((x for x in summary["tracks"] if x["id"] == tid), None)
                from PIL import Image as PILImage
                w0, h0 = PILImage.open(out_dir / "snapshots" / s).size
                width = 56 * mm
                row.append([Image(str(out_dir / "snapshots" / s), width, width * h0 / w0),
                            Paragraph(f"#{tid} <b>{tr['class']}</b> {tr['max_conf']:.2f}" if tr else s, sub)])
                if len(row) == 3:
                    cells.append(row)
                    row = []
            if row:
                cells.append(row + [""] * (3 - len(row)))
            tb = Table(cells, colWidths=[60 * mm] * 3)
            tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4)]))
            story.append(tb)
    story += [Spacer(1, 10), Paragraph("Generated by VisionInspect. Counts are tracker-merged; confidence is the mean over frames a defect was visible.", sub)]
    doc.build(story)
