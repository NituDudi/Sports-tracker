# 🏆 Sports Tracker — Streamlit App

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Usage

1. Paste any public sports video URL in the input box
2. Adjust settings in the sidebar (model, tracker, device, etc.)
3. Click **▶ Run**
4. View & download results across 5 tabs:
   - 🎬 Annotated video
   - 🌡️ Heatmap
   - 🏃 Trajectories
   - 🗺️ Bird's-eye view
   - 📈 Count chart

## Supported URLs

Anything yt-dlp supports — YouTube, Vimeo, Wikimedia, direct .mp4 links, and 250+ other sites.

## Tips

- On CPU: use `yolov8n.pt` + `every_n=2` for faster processing
- On GPU: use `yolov8l.pt` + `device=cuda` for best accuracy
- Use **Max frames** to test on just a clip of the video first
