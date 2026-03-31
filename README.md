# 🏆 Sports Tracker — Streamlit App
## 🎥 Video Source
- **Title:** Soccer Match Clip  
- **Source:** [YouTube](https://www.youtube.com/watch?v=vuCTH7jjPnw)
## 🌐 Live Demo
You can view the deployed project here:
- 🔗 [Live Host Link](YOUR_LIVE_LINK_HERE)

A Streamlit application for multi-object detection and persistent ID tracking in sports or event footage.

## Features

- Upload a local video file (`.mp4`, `.avi`, `.mov`) or paste a direct `.mp4` link
- Run YOLOv8 detection with persistent tracking via ByteTrack or BoT-SORT
- Generate:
  - annotated video with bounding boxes, IDs, and trajectory tails
  - movement heatmap
  - trajectory overlay
  - bird's-eye / top-view projection
  - count-over-time CSV
  - active-subject charts in the UI
- Tune speed and accuracy from the sidebar:
  - YOLO model variant
  - device: `cpu`, `cuda`, or `mps`
  - confidence threshold
  - tracker type
  - track buffer
  - process every N-th frame
  - max frames limit
  - trajectory tails and heatmap toggles

## Setup

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How it works

1. Load a video from upload or from a direct `.mp4` URL.
2. Load the selected YOLOv8 model.
3. Track only the `person` class (`classes=[0]`).
4. Keep track IDs alive with `persist=True` and the selected tracker.
5. Write the annotated video and save the analytics images/CSV.

## Sidebar controls

| Control | Purpose |
|---|---|
| YOLO variant | Choose between `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, and `yolov8l.pt` |
| Device | Run on `cpu`, `cuda`, or `mps` |
| Confidence threshold | Filter weak detections |
| Tracker | Switch between `bytetrack` and `botsort` |
| Track buffer | Keep disappeared tracks alive for a while |
| Process every N-th frame | Speed up processing by skipping frames |
| Max frames | Process only part of a long video |
| Trajectory tails | Show short track history on the output |
| Heatmap | Generate a cumulative activity heatmap |

## Output files

The app saves these files in the output folder:

- `annotated_output.mp4`
- `heatmap.png`
- `trajectories.png`
- `topview.png`
- `count_over_time.csv`

## Notes

- The app is optimized for person tracking in sports footage.
- `ByteTrack` is the default tracker in the UI.
- `yolov8m.pt` is the default model in the current interface.
- The chart tab shows active subject count over time and a rolling peak view.

## Tech stack

- Streamlit
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib
