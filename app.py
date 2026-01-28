import streamlit as st  # Web App
from PIL import Image   # Image Processing
import numpy as np      # Image Processing
import easyocr as ocr   # OCR

# -----------------------------
# Page config & title
# -----------------------------
st.set_page_config(page_title="Easy OCR - Extract Text from Images", page_icon="🔎", layout="centered")

st.title("Easy OCR - Extract Text from Images")
st.markdown("## Optical Character Recognition - Using `easyocr`, `streamlit` (works on 🤗 Spaces)")

st.markdown(
    "Link to the demo app - "
    "[image-to-text-app on 🤗 Spaces](https://huggingface.co/spaces/Amrrs/image-to-text-app)"
)

# -----------------------------
# Sidebar options
# -----------------------------
with st.sidebar:
    st.header("Options")
    show_boxes = st.checkbox("Show bounding boxes", value=False, help="Draw boxes around detected text on the image")
    show_conf = st.checkbox("Show confidence scores", value=True)
    # If you have a GPU available and EasyOCR compiled for it:
    use_gpu = st.checkbox("Use GPU (if available)", value=False)

# -----------------------------
# Cache the OCR Reader (deprec. fix: use st.cache_resource instead of st.cache)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_reader(langs=("en",), gpu=False):
    # model_storage_directory can help on platforms with read-only home dirs (e.g., Spaces)
    return ocr.Reader(list(langs), gpu=gpu, model_storage_directory=".")

reader = load_reader(langs=("en",), gpu=use_gpu)

# -----------------------------
# Image uploader
# -----------------------------
image_file = st.file_uploader(label="Upload your image here", type=["png", "jpg", "jpeg"])

if image_file is not None:
    input_image = Image.open(image_file).convert("RGB")  # ensure RGB mode
    st.image(input_image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🤖 Extracting text..."):
        np_img = np.array(input_image)
        # detail=1 returns bounding boxes and confidence
        result = reader.readtext(np_img, detail=1)

    # Collect text lines
    lines = [entry[1] for entry in result]  # (bbox, text, conf) -> text is entry[1]

    st.subheader("Extracted Text")
    if lines:
        st.write("\n".join(lines))
    else:
        st.info("No text detected.")

    # Show confidence table (optional)
    if show_conf and result:
        st.subheader("Detections (with confidence)")
        conf_rows = [
            {
                "text": entry[1],
                "confidence": float(entry[2]),
                "box": entry[0],
            }
            for entry in result
        ]
        st.dataframe(conf_rows, use_container_width=True)

    # Optionally draw boxes
    if show_boxes and result:
        import cv2

        boxed = np_img.copy()
        for bbox, text, conf in result:
            # bbox: list of 4 points [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(boxed, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # Put text label near the first point
            label = text if not show_conf else f"{text} ({conf:.2f})"
            x, y = pts[0]
            cv2.putText(boxed, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.subheader("Detections Preview")
        st.image(boxed, caption="Detected boxes", use_container_width=True)

    st.balloons()  # still supported
else:
    st.write("Upload an image to begin.")

st.caption("Made with ❤️ by @1littlecoder. Credits to 🤗 Spaces for hosting this.")
