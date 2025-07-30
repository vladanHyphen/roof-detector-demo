import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

st.set_page_config(page_title="Roof Detector Demo", layout="wide")

st.title("AI Roof Detector — Informal Settlement Demo")
st.markdown(
    "Upload a high-resolution satellite or aerial image, "
    "detect all dwelling units (roofs), and download an Excel with coordinates."
)

uploaded_file = st.file_uploader(
    "Upload an aerial or satellite image (PNG, JPG, JPEG)", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_rgb = np.array(image.convert('RGB'))
    st.image(img_rgb, caption="Uploaded image", use_container_width=True)

    # --- HSV mask for gray-like roofs ---
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    roof_mask = ((s < 65) & (v > 100)).astype(np.uint8) * 255

    # --- Morphological cleaning ---
    kernel = np.ones((2, 2), np.uint8)
    roof_mask_clean = cv2.morphologyEx(roof_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- Find contours and centroids ---
    contours, _ = cv2.findContours(roof_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_result = img_rgb.copy()
    centroids = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
                cv2.circle(overlay_result, (cx, cy), 8, (0,255,0), -1)  # green dot

    num_dwellings = len(centroids)
    st.success(f"Detected {num_dwellings} dwellings (green dots)")

    # --- Annotate image ---
    cv2.putText(
        overlay_result,
        f'Dwellings detected: {num_dwellings}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255,255,255),
        3,
        cv2.LINE_AA
    )
    cv2.putText(
        overlay_result,
        f'Dwellings detected: {num_dwellings}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0,0,0),
        1,
        cv2.LINE_AA
    )

    st.image(overlay_result, caption="Detected dwellings (green dots)", use_container_width=True)

    # --- Prepare Excel file for download ---
    df = pd.DataFrame([
        {'Dwelling_ID': idx+1, 'X_pixel': cx, 'Y_pixel': cy, 'Address': f'Dwelling_{idx+1}'}
        for idx, (cx, cy) in enumerate(centroids)
    ])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)

    st.download_button(
        label="Download Excel with Dwelling Coordinates",
        data=output,
        file_name="roof_centroids.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


