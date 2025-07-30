import streamlit as st

st.title("Roof Detector Demo — Image Upload Test")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
    st.success("Image uploaded successfully!")

