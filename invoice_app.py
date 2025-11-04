import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import json
import os

# Import your actual utils
from invoice_utils import preprocess_image, extract_invoice_fields

st.set_page_config(page_title="📄 Invoice Field Extractor", layout="centered")
st.title("📄 Invoice Data Extractor")

st.write("Upload an invoice image below. The system will extract key fields and return them in JSON format.")

# ---- File Uploader ----
uploaded_file = st.file_uploader("📤 Upload Invoice Image", type=["png", "jpg", "jpeg", "tiff"])

if uploaded_file:
    # Display the uploaded image
    st.image(uploaded_file, caption="📄 Uploaded Invoice", use_container_width=True)

    # Convert to NumPy array for OpenCV
    image_np = np.array(Image.open(uploaded_file))
    processed = preprocess_image(image_np, from_file=False)
    pil_img = Image.fromarray(processed)
    text = pytesseract.image_to_string(pil_img, lang="eng")
    extracted = extract_invoice_fields(text, pil_img)

    # ---- Show Output ----
    st.subheader("✅ Extracted Invoice Data")
    st.json(extracted)  # pretty JSON output

    # ---- Optional: Download JSON ----
    st.download_button(
        label="💾 Download JSON",
        data=json.dumps(extracted, indent=4),
        file_name="invoice_extracted.json",
        mime="application/json"
    )
