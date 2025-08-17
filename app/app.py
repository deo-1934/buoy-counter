import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# ---------- Page ----------
st.set_page_config(page_title="Buoy Counter (YOLOv8)", layout="wide")
st.title("🌊 Buoy Counter")
st.caption("Upload an image → Tiled YOLOv8 inference → Count buoys")

# ---------- Fixed settings ----------
MODEL_PATH = "runs/detect/buoy_yolov8n_v12/weights/best.pt"  # adjust if your run name differs
CONF = 0.03
IOU  = 0.55
TILE = 1024       # tile size (pixels)
OVERLAP = 0.20    # 20% overlap between tiles
NMS_IOU = 0.5     # NMS for merging tile results

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

model = load_model(MODEL_PATH)

def iou_xyxy(a, b):
    # a, b: [x1, y1, x2, y2]
    inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def nms(boxes, scores, iou_thr=0.5):
    # boxes: Nx4, scores: N
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou_xyxy(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious <= iou_thr]
    return keep

def predict_tiled(rgb: np.ndarray):
    H, W, _ = rgb.shape
    step = int(TILE * (1 - OVERLAP))
    all_boxes, all_scores = [], []

    for y in range(0, H, step):
        for x in range(0, W, step):
            y2 = min(y + TILE, H)
            x2 = min(x + TILE, W)
            tile = rgb[y:y2, x:x2]
            r = model.predict(source=tile, imgsz=TILE, conf=CONF, iou=IOU, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            # offset to global coords
            boxes[:, [0, 2]] += x
            boxes[:, [1, 3]] += y
            all_boxes.append(boxes)
            all_scores.append(scores)

    if not all_boxes:
        return np.zeros((0, 4)), np.array([])

    boxes = np.vstack(all_boxes)
    scores = np.hstack(all_scores)
    keep = nms(boxes, scores, iou_thr=NMS_IOU)
    return boxes[keep], scores[keep]

# ---------- UI ----------
uploaded = st.file_uploader("Upload an image…", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.info("Please upload an image to run the model.")
    st.stop()

img_pil = Image.open(uploaded).convert("RGB")
img_np = np.array(img_pil)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Input Image")
    st.image(img_pil, use_container_width=True)

with st.spinner("Running tiled prediction..."):
    boxes, scores = predict_tiled(img_np)

count = int(len(boxes))
out = img_pil.copy()
draw = ImageDraw.Draw(out)
for (x1, y1, x2, y2) in boxes:
    draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=(0, 255, 0), width=2)

with col2:
    st.subheader(f"Detected buoys: {count}")
    st.image(out, caption="Model Output", use_container_width=True)

with st.expander("Inference settings"):
    st.write({
        "tiling": f"{TILE}px, overlap {int(OVERLAP*100)}%",
        "conf": CONF, "iou": IOU, "merge_nms_iou": NMS_IOU
    })
