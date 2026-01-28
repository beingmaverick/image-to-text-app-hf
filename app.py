# app_gradio.py
import gradio as gr
import numpy as np
from PIL import Image
import easyocr as ocr

# Optional import for drawing bounding boxes
import cv2

# Cache the OCR reader for performance (Gradio has its own caching via global persistence)
_reader_cache = {}

def get_reader(langs=("en",), use_gpu=False):
    key = (tuple(langs), use_gpu)
    if key not in _reader_cache:
        _reader_cache[key] = ocr.Reader(list(langs), gpu=use_gpu, model_storage_directory=".")
    return _reader_cache[key]

def run_ocr(image: Image.Image, show_boxes: bool, show_conf: bool, use_gpu: bool, languages: str):
    """
    image: PIL image from Gradio
    show_boxes: bool to render bounding boxes
    show_conf: bool to include confidence in output
    use_gpu: bool to initialize OCR reader with GPU
    languages: comma-separated string like 'en,fr'
    """
    if image is None:
        return None, "Please upload an image.", None

    langs = [l.strip() for l in languages.split(",") if l.strip()]
    if not langs:
        langs = ["en"]

    reader = get_reader(langs=tuple(langs), use_gpu=use_gpu)

    # Ensure RGB
    image = image.convert("RGB")
    np_img = np.array(image)

    # detail=1 gives (bbox, text, conf)
    result = reader.readtext(np_img, detail=1)

    # Extract text lines
    lines = [entry[1] for entry in result]

    # Prepare textual output
    if not lines:
        text_output = "No text detected."
    else:
        if show_conf:
            text_output = "\n".join([f"{entry[1]} (conf: {float(entry[2]):.2f})" for entry in result])
        else:
            text_output = "\n".join(lines)

    # Optionally draw bounding boxes
    boxed_preview = None
    if show_boxes and result:
        boxed = np_img.copy()
        for bbox, text, conf in result:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(boxed, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            label = text if not show_conf else f"{text} ({conf:.2f})"
            x, y = pts[0]
            cv2.putText(boxed, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        boxed_preview = Image.fromarray(boxed)

    # Return: original image, textual result, boxed image (optional)
    return image, text_output, boxed_preview


with gr.Blocks(title="Easy OCR - Gradio") as demo:
    gr.Markdown("# 🔎 Easy OCR - Extract Text from Images (Gradio)")
    gr.Markdown("**Optical Character Recognition** using `easyocr` + `gradio`. Upload an image to extract text.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload your image", sources=["upload", "clipboard"])
            languages = gr.Textbox(value="en", label="Languages (comma-separated)", info="e.g., en,fr,de")
            show_boxes = gr.Checkbox(value=False, label="Show bounding boxes")
            show_conf = gr.Checkbox(value=True, label="Show confidence scores")
            use_gpu = gr.Checkbox(value=False, label="Use GPU (if available)")

            run_btn = gr.Button("Run OCR")

        with gr.Column(scale=1):
            original_preview = gr.Image(label="Original Image", interactive=False)
            text_output = gr.Textbox(label="Extracted Text", lines=8)
            boxes_preview = gr.Image(label="Detections (boxes)", interactive=False)

    run_btn.click(
        fn=run_ocr,
        inputs=[image_input, show_boxes, show_conf, use_gpu, languages],
        outputs=[original_preview, text_output, boxes_preview],
    )

    gr.Markdown("Made with ❤️ using Gradio and EasyOCR")

if __name__ == "__main__":
    demo.launch()
