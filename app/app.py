import streamlit as st

st.set_page_config(page_title="Buoy Counter", layout="wide")

st.title("🌊 Buoy Counter")
st.write("Upload an image and let the AI count the buoys for you.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("👉 Model prediction will be shown here soon...")
