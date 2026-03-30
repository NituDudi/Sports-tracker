import streamlit as st
import cv2
import numpy as np
import os
import subprocess
import sys
import csv
import time
import tempfile
import shutil
from collections import defaultdict, deque
from pathlib import Path
import threading

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sports Tracker",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark sporty theme */
  .stApp { background: #0d1117; }
  section[data-testid="stSidebar"] { background: #161b22; }
  section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

  /* Main text */
  .stApp, .stMarkdown, .stText { color: #e6edf3; }

  /* Title banner */
  .title-banner {
    background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 28px 36px 22px;
    margin-bottom: 24px;
  }
  .title-banner h1 { color: #58a6ff; margin: 0 0 6px; font-size: 2rem; }
  .title-banner p  { color: #8b949e; margin: 0; font-size: 0.95rem; }

  /* Metric card */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    margin: 4px;
  }
  .metric-card .val { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .metric-card .lbl { font-size: 0.78rem; color: #8b949e; margin-top: 2px; text-transform: uppercase; letter-spacing: .05em; }

  /* Status badge */
  .badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 20px; font-size: 0.8rem; font-weight: 600;
  }
  .badge-running  { background:#1f4329; color:#3fb950; border:1px solid #238636; }
  .badge-done     { background:#0d2f4f; color:#58a6ff; border:1px solid #1f6feb; }
  .badge-error    { background:#4a1616; color:#f85149; border:1px solid #da3633; }

  /* Progress area */
  .log-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #7ee787;
    max-height: 220px;
    overflow-y: auto;
  }

  /* Section headers */
  .section-header {
    color: #8b949e;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin: 20px 0 12px;
  }

  /* Image caption */
  .img-caption {
    text-align: center;
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 6px;
  }

  /* Buttons */
  .stButton > button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    width: 100%;
  }
  .stButton > button:hover { background: #2ea043 !important; }

  /* Download button */
  .stDownloadButton > button {
    background: #1f6feb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%;
  }

  /* Input boxes */
  .stTextInput input, .stSelectbox select {
    background: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
  }

  /* Slider */
  .stSlider .rc-slider-track { background: #238636 !important; }
  .stSlider .rc-slider-handle { border-color: #238636 !important; }

  /* Tab */
  .stTabs [role="tab"] { color: #8b949e; }
  .stTabs [role="tab"][aria-selected="true"] { color: #58a6ff; border-color: #58a6ff; }

  /* Progress bar */
  .stProgress .st-bo { background: #238636; }

  div[data-testid="stVerticalBlock"] { gap: 0.5rem; }
  div[data-testid="column"] { padding: 0 6px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = [
    (255, 56, 56),  (255,157, 51), (255,247, 51), ( 51,255, 51),
    ( 51,255,255),  ( 51,148,255), (148, 51,255), (255, 51,255),
    (255,153,153),  (153,255,153), (153,153,255), (255,228,153),
    (153,255,228),  (228,153,255), (255,200,100), (100,255,200),
    (200,100,255),  (255,100,200), (100,200,255), (200,255,100),
]
def id_color(tid): return PALETTE[int(tid) % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def draw_box(frame, x1, y1, x2, y2, tid, conf):
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    color = id_color(tid)
    ov = frame.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),color,-1)
    cv2.addWeighted(ov,0.12,frame,0.88,0,frame)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    text = f"ID:{tid}  {conf:.2f}"
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.55,2)
    p=4; lx1,ly1,lx2,ly2 = x1,max(0,y1-th-2*p),x1+tw+2*p,y1
    cv2.rectangle(frame,(lx1,ly1),(lx2,ly2),color,-1)
    cv2.putText(frame,text,(lx1+p,ly2-p),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)

def draw_trajectory(frame, points, color, max_len=60):
    pts = list(points)[-max_len:]
    for i in range(1,len(pts)):
        alpha = i/len(pts)
        faded = tuple(int(c*alpha) for c in color)
        cv2.line(frame,pts[i-1],pts[i],faded,2,cv2.LINE_AA)
    if pts: cv2.circle(frame,pts[-1],4,color,-1,cv2.LINE_AA)

def draw_hud(frame, count, frame_idx, total):
    cv2.rectangle(frame,(8,8),(280,68),(15,15,15),-1)
    cv2.rectangle(frame,(8,8),(280,68),(80,80,80),1)
    cv2.putText(frame,f"Active subjects: {count}",(16,38),
                cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2,cv2.LINE_AA)
    prog = frame_idx/max(total-1,1)
    cv2.rectangle(frame,(12,50),(265,62),(50,50,50),-1)
    cv2.rectangle(frame,(12,50),(12+int(253*prog),62),(0,210,0),-1)
    cv2.putText(frame,f"{int(prog*100)}%",(268,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,180),1,cv2.LINE_AA)

def add_to_heatmap(heat, fx, fy, r=22):
    fx,fy = int(fx),int(fy)
    if 0<=fx<heat.shape[1] and 0<=fy<heat.shape[0]:
        cv2.circle(heat,(fx,fy),r,1.0,-1)

def render_heatmap(heat, background=None, alpha=0.60):
    norm = cv2.normalize(heat,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    col  = cv2.applyColorMap(norm,cv2.COLORMAP_JET)
    if background is not None:
        bg = cv2.resize(background,(heat.shape[1],heat.shape[0]))
        return cv2.addWeighted(bg,1-alpha,col,alpha,0)
    return col

def build_topview(trajs, frame_size, canvas_size=(640,420)):
    fW,fH = frame_size; cW,cH = canvas_size
    canvas = np.full((cH,cW,3),25,dtype=np.uint8)
    # pitch markings
    cv2.rectangle(canvas,(20,20),(cW-20,cH-20),(60,60,60),1)
    cv2.line(canvas,(cW//2,20),(cW//2,cH-20),(60,60,60),1)
    cv2.circle(canvas,(cW//2,cH//2),60,(60,60,60),1)
    sx,sy = cW/fW, cH/fH
    for tid,pts in trajs.items():
        color  = id_color(tid)
        scaled = [(int(x*sx),int(y*sy)) for x,y in pts]
        for i in range(1,len(scaled)):
            cv2.line(canvas,scaled[i-1],scaled[i],color,1,cv2.LINE_AA)
        if scaled: cv2.circle(canvas,scaled[-1],5,color,-1)
    cv2.putText(canvas,"Bird's-eye view",(8,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(130,130,130),1)
    return canvas

def build_trajectory_img(traj_canvas, first_frame):
    dark = (first_frame*0.22).astype(np.uint8)
    return cv2.addWeighted(dark,0.5,traj_canvas,0.85,0)


# ══════════════════════════════════════════════════════════════════════════════
# INSTALL / DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model(model_name, device):
    try:
        from ultralytics import YOLO
        return YOLO(model_name)
    except ImportError:
        subprocess.run([sys.executable,"-m","pip","install","ultralytics","-q"],check=True)
        from ultralytics import YOLO
        return YOLO(model_name)

def ensure_ytdlp():
    try:
        subprocess.run(["yt-dlp","--version"],capture_output=True,check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run([sys.executable,"-m","pip","install","yt-dlp","-q"],check=True)

def download_video(url, out_path, log_fn=None):
    if log_fn: log_fn("⬇️ Downloading video...")

    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", url,
         "-f", "best[ext=mp4]",
         "-o", out_path,
         "--no-playlist"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    if log_fn: log_fn("✅ Download complete")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_pipeline(
    video_path, model, cfg, out_dir,
    progress_cb=None, log_cb=None, cancel_event=None
):
    """
    Full detection + tracking + visualisation pipeline.
    Returns dict of output file paths + stats.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Video meta ─────────────────────────────────────────────────────────
    cap   = cv2.VideoCapture(video_path)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if log_cb: log_cb(f"📹  Video: {W}×{H} @ {FPS:.1f}fps | {TOTAL} frames")

    out_fps   = FPS / cfg["every_n"]
    ann_path  = os.path.join(out_dir, "annotated_output.mp4")
    heat_path = os.path.join(out_dir, "heatmap.png")
    traj_path = os.path.join(out_dir, "trajectories.png")
    tv_path   = os.path.join(out_dir, "topview.png")
    csv_path  = os.path.join(out_dir, "count_over_time.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(ann_path, fourcc, out_fps, (W, H))

    # State
    traj_history = defaultdict(lambda: deque(maxlen=cfg["track_buffer"]*2))
    heat_acc     = np.zeros((H,W), dtype=np.float32)
    traj_canvas  = np.zeros((H,W,3), dtype=np.uint8)
    count_rows   = []
    first_frame  = None
    frame_idx    = 0
    processed    = 0
    t0           = time.time()

    cap = cv2.VideoCapture(video_path)

    while True:
        if cancel_event and cancel_event.is_set():
            if log_cb: log_cb("⛔ Cancelled by user")
            break

        ok, frame = cap.read()
        if not ok: break

        if frame_idx % cfg["every_n"] != 0:
            frame_idx += 1
            continue

        if first_frame is None:
            first_frame = frame.copy()

        # ── Track ───────────────────────────────────────────────────────────
        results = model.track(
            source  = frame,
            persist = True,
            tracker = cfg["tracker"],
            conf    = cfg["conf"],
            iou     = cfg["iou"],
            classes = cfg["classes"],
            imgsz   = cfg["imgsz"],
            device  = cfg["device"],
            verbose = False,
        )

        out_frame    = frame.copy()
        active_count = 0

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None: continue
            ids   = boxes.id.int().tolist()
            xyxys = boxes.xyxy.tolist()
            confs = boxes.conf.tolist()

            # Trajectory tails first
            if cfg["draw_traj"]:
                for tid in ids:
                    draw_trajectory(out_frame, traj_history[tid],
                                    id_color(tid), cfg["traj_len"])
                    draw_trajectory(traj_canvas, traj_history[tid],
                                    id_color(tid), cfg["traj_len"])

            for tid, xyxy, conf in zip(ids, xyxys, confs):
                x1,y1,x2,y2 = xyxy
                cx,cy = int((x1+x2)/2), int((y1+y2)/2)
                traj_history[tid].append((cx,cy))
                if cfg["draw_heatmap"]:
                    add_to_heatmap(heat_acc, int((x1+x2)/2), int(y2))
                draw_box(out_frame, x1, y1, x2, y2, tid, conf)
                active_count += 1

        draw_hud(out_frame, active_count, frame_idx, TOTAL)
        writer.write(out_frame)
        count_rows.append({"frame": frame_idx,
                            "time_s": round(frame_idx/FPS,3),
                            "count": active_count})

        frame_idx += 1
        processed += 1

        # Progress callback
        if progress_cb:
            pct = processed / max(cfg["max_frames"] or TOTAL, 1)
            elapsed = time.time()-t0
            eta = (elapsed/max(processed,1))*(max(cfg["max_frames"] or TOTAL,1)-processed)
            progress_cb(min(pct,1.0), processed, active_count, eta)

        if log_cb and processed % 60 == 0:
            pct = 100*processed/max(cfg["max_frames"] or TOTAL,1)
            log_cb(f"🎬  Frame {frame_idx}/{TOTAL} ({pct:.0f}%) — active: {active_count}")

        if cfg["max_frames"] and processed >= cfg["max_frames"]:
            break

    cap.release()
    writer.release()
    if log_cb: log_cb(f"✅ Annotated video saved")

    # ── Post-process ─────────────────────────────────────────────────────
    heatmap = render_heatmap(heat_acc, background=first_frame)
    cv2.imwrite(heat_path, heatmap)

    traj_out = build_trajectory_img(traj_canvas, first_frame)
    cv2.imwrite(traj_path, traj_out)

    tv = build_topview(traj_history, (W, H))
    cv2.imwrite(tv_path, tv)

    with open(csv_path,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame","time_s","count"])
        w.writeheader(); w.writerows(count_rows)

    if log_cb: log_cb("✅ All outputs saved")

    # ── Stats ─────────────────────────────────────────────────────────────
    lengths = [len(v) for v in traj_history.values()]
    counts  = [r["count"] for r in count_rows]
    stats = {
        "total_ids":    len(lengths),
        "peak_count":   max(counts) if counts else 0,
        "mean_count":   round(sum(counts)/len(counts),1) if counts else 0,
        "mean_traj":    round(sum(lengths)/len(lengths),1) if lengths else 0,
        "max_traj":     max(lengths) if lengths else 0,
        "frames_proc":  processed,
        "duration_s":   round(time.time()-t0,1),
    }
    return {
        "annotated": ann_path,
        "heatmap":   heat_path,
        "trajectories": traj_path,
        "topview":   tv_path,
        "csv":       csv_path,
        "stats":     stats,
        "count_rows": count_rows,
        "first_frame": first_frame,
        "W": W, "H": H,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="section-header">Model</div>', unsafe_allow_html=True)
    model_name = st.selectbox("YOLO variant",
        ["yolov8n.pt","yolov8s.pt","yolov8m.pt","yolov8l.pt"],
        index=2,
        help="n=fastest, m=balanced, l=most accurate")
    device = st.selectbox("Device", ["cpu","cuda","mps"], index=0)
    conf   = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)

    st.markdown('<div class="section-header">Tracker</div>', unsafe_allow_html=True)
    tracker    = st.selectbox("Tracker", ["bytetrack","botsort"], index=0)
    trk_buffer = st.slider("Track buffer (frames)", 10, 90, 30)

    st.markdown('<div class="section-header">Processing</div>', unsafe_allow_html=True)
    every_n    = st.slider("Process every N-th frame", 1, 5, 1,
                           help="Higher = faster but less smooth")
    max_frames = st.number_input("Max frames (0 = full video)", 0, 10000, 0)

    st.markdown('<div class="section-header">Visualisation</div>', unsafe_allow_html=True)
    draw_traj  = st.toggle("Trajectory tails", value=True)
    traj_len   = st.slider("Tail length (frames)", 10, 120, 60)
    draw_heat  = st.toggle("Heatmap", value=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="title-banner">
  <h1>🏆 Sports Multi-Object Tracker</h1>
  <p>Paste any public sports video URL → get annotated output, heatmap, trajectories &amp; bird's-eye view</p>
</div>
""", unsafe_allow_html=True)

# ── URL input ────────────────────────────────────────────────────────────────
col_url, col_btn = st.columns([5,1])
with col_url:
    video_url = st.text_input(
        "Video URL",
        placeholder="https://www.youtube.com/watch?v=...   or any direct .mp4 link",
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("▶  Run", use_container_width=True)

st.caption("Works with YouTube, Vimeo, Wikimedia, direct MP4 links, and [250+ other sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md) via yt-dlp")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
if run_clicked:
    if not video_url.strip():
        st.warning("Please enter a video URL first.")
        st.stop()

    cfg = {
        "tracker":     f"{tracker}.yaml",
        "conf":        conf,
        "iou":         0.45,
        "classes":     [0],
        "imgsz":       640,
        "device":      device,
        "every_n":     every_n,
        "max_frames":  int(max_frames) if max_frames > 0 else None,
        "track_buffer": trk_buffer,
        "draw_traj":   draw_traj,
        "traj_len":    traj_len,
        "draw_heatmap": draw_heat,
    }

    # Working directory
    work_dir = tempfile.mkdtemp(prefix="sports_tracker_")
    vid_path = os.path.join(work_dir, "input_video.mp4")

    # UI elements
    status_box  = st.empty()
    prog_bar    = st.progress(0.0)
    stats_row   = st.empty()
    log_box     = st.empty()
    log_lines   = []

    def log(msg):
        log_lines.append(msg)
        log_box.markdown(
            "<div class='log-box'>" +
            "<br>".join(log_lines[-12:]) +
            "</div>", unsafe_allow_html=True
        )

    def progress(pct, processed, active, eta):
        prog_bar.progress(float(pct))
        stats_row.markdown(f"""
        <div style="display:flex;gap:8px;margin:6px 0">
          <div class="metric-card" style="flex:1"><div class="val">{processed}</div><div class="lbl">Frames done</div></div>
          <div class="metric-card" style="flex:1"><div class="val">{active}</div><div class="lbl">Active IDs</div></div>
          <div class="metric-card" style="flex:1"><div class="val">{int(pct*100)}%</div><div class="lbl">Progress</div></div>
          <div class="metric-card" style="flex:1"><div class="val">{int(eta)}s</div><div class="lbl">ETA</div></div>
        </div>""", unsafe_allow_html=True)

    try:
        # Step 1 — Download
        status_box.markdown('<span class="badge badge-running">⏳ Downloading video…</span>', unsafe_allow_html=True)
        download_video(video_url, vid_path, log_fn=log)

        # Step 2 — Load model
        status_box.markdown('<span class="badge badge-running">⏳ Loading model…</span>', unsafe_allow_html=True)
        log(f"🤖  Loading {model_name} on {device}…")
        model = load_model(model_name, device)
        log(f"✅  Model ready")

        # Step 3 — Pipeline
        status_box.markdown('<span class="badge badge-running">⏳ Tracking…</span>', unsafe_allow_html=True)
        log("🎬  Starting tracking pipeline…")

        outputs = run_pipeline(
            vid_path, model, cfg,
            out_dir     = os.path.join(work_dir, "outputs"),
            progress_cb = progress,
            log_cb      = log,
        )

        prog_bar.progress(1.0)
        status_box.markdown('<span class="badge badge-done">✅ Done!</span>', unsafe_allow_html=True)
        log("🎉  Pipeline complete!")

        # ── Store in session state ────────────────────────────────────────
        st.session_state["outputs"]  = outputs
        st.session_state["work_dir"] = work_dir

    except Exception as e:
        status_box.markdown(f'<span class="badge badge-error">❌ Error</span>', unsafe_allow_html=True)
        log(f"❌  {str(e)[:300]}")
        st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════════════════
if "outputs" in st.session_state:
    outputs = st.session_state["outputs"]
    stats   = outputs["stats"]

    st.markdown("---")
    st.markdown("### 📊 Results")

    # ── Stats cards ──────────────────────────────────────────────────────────
    c1,c2,c3,c4,c5 = st.columns(5)
    cards = [
        (c1, stats["total_ids"],    "Unique IDs"),
        (c2, stats["peak_count"],   "Peak count"),
        (c3, stats["mean_count"],   "Avg count"),
        (c4, stats["frames_proc"],  "Frames"),
        (c5, f'{stats["duration_s"]}s', "Runtime"),
    ]
    for col, val, lbl in cards:
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_vid, tab_heat, tab_traj, tab_top, tab_chart = st.tabs([
        "🎬 Annotated Video",
        "🌡️ Heatmap",
        "🏃 Trajectories",
        "🗺️ Bird's-eye",
        "📈 Count Chart",
    ])

    with tab_vid:
        if os.path.exists(outputs["annotated"]):
            with open(outputs["annotated"],"rb") as f:
                vid_bytes = f.read()
            st.video(vid_bytes)
            st.markdown('<div class="img-caption">Annotated output — bounding boxes, unique IDs, trajectory tails</div>', unsafe_allow_html=True)
            st.download_button("⬇️ Download annotated video",
                               data=vid_bytes, file_name="annotated_output.mp4",
                               mime="video/mp4")

    with tab_heat:
        if os.path.exists(outputs["heatmap"]):
            img = cv2.imread(outputs["heatmap"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, use_container_width=True)
            st.markdown('<div class="img-caption">Cumulative movement heatmap — warmer = more activity</div>', unsafe_allow_html=True)
            with open(outputs["heatmap"],"rb") as f:
                st.download_button("⬇️ Download heatmap", data=f.read(),
                                   file_name="heatmap.png", mime="image/png")

    with tab_traj:
        if os.path.exists(outputs["trajectories"]):
            img = cv2.imread(outputs["trajectories"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, use_container_width=True)
            st.markdown('<div class="img-caption">All trajectories — each colour = one unique ID</div>', unsafe_allow_html=True)
            with open(outputs["trajectories"],"rb") as f:
                st.download_button("⬇️ Download trajectories", data=f.read(),
                                   file_name="trajectories.png", mime="image/png")

    with tab_top:
        if os.path.exists(outputs["topview"]):
            img = cv2.imread(outputs["topview"])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, use_container_width=True)
            st.markdown('<div class="img-caption">Bird\'s-eye view — all track paths projected onto a pitch canvas</div>', unsafe_allow_html=True)
            with open(outputs["topview"],"rb") as f:
                st.download_button("⬇️ Download top-view", data=f.read(),
                                   file_name="topview.png", mime="image/png")

    with tab_chart:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = outputs["count_rows"]
        if rows:
            xs = [r["time_s"] for r in rows]
            ys = [r["count"]  for r in rows]

            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            fig.patch.set_facecolor("#0d1117")
            for ax in axes:
                ax.set_facecolor("#161b22")
                ax.tick_params(colors="#8b949e")
                for s in ax.spines.values(): s.set_color("#30363d")

            axes[0].plot(xs, ys, color="#58a6ff", linewidth=1.4)
            axes[0].fill_between(xs, ys, alpha=0.12, color="#58a6ff")
            axes[0].set_xlabel("Time (s)", color="#8b949e")
            axes[0].set_ylabel("Active subjects", color="#8b949e")
            axes[0].set_title("Active tracked subjects over time", color="#e6edf3")
            axes[0].grid(True, alpha=0.15)

            # Rolling max track lengths — build from count_rows
            window = max(1, len(xs)//20)
            rolled = [max(ys[max(0,i-window):i+1]) for i in range(len(ys))]
            axes[1].plot(xs, rolled, color="#3fb950", linewidth=1.4)
            axes[1].fill_between(xs, rolled, alpha=0.12, color="#3fb950")
            axes[1].set_xlabel("Time (s)", color="#8b949e")
            axes[1].set_ylabel("Max active (rolling)", color="#8b949e")
            axes[1].set_title("Rolling peak count", color="#e6edf3")
            axes[1].grid(True, alpha=0.15)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # CSV download
            with open(outputs["csv"],"rb") as f:
                st.download_button("⬇️ Download count CSV", data=f.read(),
                                   file_name="count_over_time.csv",
                                   mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#484f58;font-size:0.8rem;'>"
    "YOLOv8 + ByteTrack · OpenCV · Streamlit"
    "</p>",
    unsafe_allow_html=True
)
