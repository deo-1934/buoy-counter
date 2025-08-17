import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ----- Page -----
st.set_page_config(page_title="Buoy Counter (YOLOv8)", layout="wide")
st.title("🌊 Buoy Counter")
st.caption("Upload an image → Auto-run YOLOv8 → Count buoys")

# ----- Fixed inference settings (tuned) -----
MODEL_PATH = "runs/detect/buoy_yolov8n_v12/weights/best.pt"  # adjust if your run name differs
CONF = 0.03
IOU = 0.55
IMGSZ = 1024

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(MODEL_PATH)

uploaded = st.file_uploader("Upload an image…", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to run the model.")
    st.stop()

img = Image.open(uploaded).convert("RGB")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Image")
    st.image(img, use_container_width=True)

with st.spinner("Running prediction..."):
    results = model.predict(
        source=np.array(img),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        augment=True,      # small boost for crowded rows
        verbose=False
    )
res = results[0]

count = len(res.boxes) if res.boxes is not None else 0
plotted = res.plot()[:, :, ::-1]  # BGR->RGB

with col2:
    st.subheader(f"Detected buoys: {int(count)}")
    st.image(plotted, caption="Model Output", use_container_width=True)

with st.expander("Inference settings"):
    st.write({"imgsz": IMGSZ, "conf": CONF, "iou": IOU, "augment": True})
