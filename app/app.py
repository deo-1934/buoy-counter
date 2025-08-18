import streamlit as st
import tempfile
import cv2
from inference_sdk import InferenceHTTPClient

# Roboflow client config
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="8J49lXK8I0S8aq9Zojqb"   # 👈 API Key تو
)

st.title("🌊 Buoy Counter App")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # نمایش تصویر آپلودی
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # ذخیره فایل آپلودی
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # مرحله resize + inference با spinner
    with st.spinner("Resizing image and detecting buoys..."):
        try:
            # خواندن تصویر
            img = cv2.imread(tmp_path)

            # تغییر سایز (YOLO معمولا 640x640 استفاده می‌کنه)
            resized_img = cv2.resize(img, (640, 640))

            # ذخیره دوباره تصویر resize شده
            resized_path = tmp_path.replace(".jpg", "_resized.jpg")
            cv2.imwrite(resized_path, resized_img)

            # پردازش تصویر
            result = client.infer(resized_path, model_id="buoy-wn6n2/2")  # 👈 model_id دقیق
            predictions = result.get("predictions", [])

            st.success("Detection completed!")
            st.write(f"**Total buoys detected:** {len(predictions)}")

            # نمایش جزئیات
            st.json(predictions)

        except Exception as e:
            st.error(f"Error during inference: {e}")
