import cv2
import os
import pytesseract
from pytesseract import Output
import json
from PIL import Image
import re

def preprocess_image(input_data, from_file=True):
    """
    input_data: either a file path (if from_file=True) or a numpy array (if from_file=False)
    """
    if from_file:
        img = cv2.imread(input_data)
    else:
        img = input_data  # already a NumPy array

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional resize (helps if text is small)
    scale_percent = 150
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    resized = cv2.resize(thresh, (width, height), interpolation=cv2.INTER_LINEAR)

    return resized



def extract_date_with_bbox(pil_img, full_text):
    """
    Try to extract the date by checking bounding boxes near the 'Date' label.
    Falls back to regex if nothing is found.
    """
    data = pytesseract.image_to_data(pil_img, output_type=Output.DICT, lang="eng")

    # Collect all detected words with positions
    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if txt:
            words.append({
                "text": txt,
                "left": data["left"][i],
                "top": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i]
            })

    # 1️⃣ Find 'Date' keyword positions
    for w in words:
        if re.search(r"^date$", w["text"], re.IGNORECASE):
            band_top = w["top"] - 10
            band_bottom = w["top"] + w["height"] + 10

            # 2️⃣ Search to the right on same horizontal band
            candidates = [
                cw["text"]
                for cw in words
                if cw["left"] > w["left"] and band_top <= cw["top"] <= band_bottom
            ]

            if candidates:
                # Join 1–3 tokens in case date is split (e.g., "12" "/" "10" "/2024")
                joined = " ".join(candidates[:3])
                # Optional: validate if it looks like a date
                if re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2}", joined):
                    return joined.strip()

    # 3️⃣ Fallback: regex over full text if bbox fails
    match = re.search(
        r'(Date(?: of issue)?[:\-]?\s*)(\d{2}[\/\-]\d{2}[\/\-]\d{4}|\d{4}[\/\-]\d{2}[\/\-]\d{2})',
        full_text,
        re.IGNORECASE
    )
    if match:
        return match.group(2)

    return None


def extract_invoice_fields(text, pil_img):
    """Main field extraction with bbox-assisted date parsing."""
    invoice_no, date, seller, tax_id, items = None, None, None, None, None

    # ✅ Invoice number
    match = re.search(r'Invoice\s*No[:\-]?\s*(\w+)', text, re.IGNORECASE)
    if match:
        invoice_no = match.group(1)

    # ✅ Seller
    match = re.search(r'Seller[:\-]?\s*(.+)', text, re.IGNORECASE)
    if match:
        seller = match.group(1).strip()

    # ✅ Tax ID
    match = re.search(r'(Tax\s*ID|Tx\s*ID)[:\-]?\s*(\w+)', text, re.IGNORECASE)
    if match:
        tax_id = match.group(2)

    # ✅ Items
    match = re.search(r'Items[:\-]?\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    if match:
        items = match.group(1).strip()

    # ✅ Date (now uses bbox if regex fails)
    date = extract_date_with_bbox(pil_img, text)

    return {
        "invoice_no": invoice_no,
        "date of issue": date,
        "seller": seller,
        "tax_id": tax_id,
        "items": items
    }



