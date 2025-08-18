import streamlit as st
import tempfile
from inference_sdk import InferenceHTTPClient
from PIL import Image
import base64

# Client config
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="8J49lXK8I0S8aq9Zojqb"
)

st.title("🌊 Buoy Counter App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # نمایش تصویر آپلودی
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # ذخیره فایل روی دیسک
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # تغییر سایز تصویر برای سرعت بهتر (حداکثر 1024 پیکسل)
    img = Image.open(tmp_path)
    img.thumbnail((1024, 1024))
    img.save(tmp_path)

    # پردازش با Roboflow
    with st.spinner("⏳ Model is processing... Please wait."):
        result = client.run_workflow(
            workspace_name="buoycounter",
            workflow_id="detect-count-and-visualize",
            images={"image": tmp_path},  
            use_cache=True
        )

    # نمایش خروجی
    st.success(f"✅ Number of buoys detected: {result[0]['count_objects']}")

    # اگر Roboflow تصویر Annotated برگردوند
    if "output_image" in result[0]:
        output_b64 = result[0]["output_image"]
        output_bytes = base64.b64decode(output_b64)
        st.image(output_bytes, caption="Detection Result", use_container_width=True)
