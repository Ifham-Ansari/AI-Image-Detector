from pathlib import Path
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import gradio as gr
import torch
from PIL import Image

from config import CHECKPOINT_DIR, DEVICE, CLASS_NAMES
from src.dataset import get_transforms
from src.gradcam import generate_gradcam
from src.predict import build_model, predict_image


def load_best_model() -> torch.nn.Module:
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return build_model(checkpoint_path, device=DEVICE)


try:
    MODEL = load_best_model()
except FileNotFoundError:
    MODEL = None


def classify_image(image: Image.Image):
    if image is None:
        return "No image", "0.00%", None
    if MODEL is None:
        return "Model not loaded", "0.00%", None

    prediction = predict_image(image, MODEL, DEVICE)
    heatmap = generate_gradcam(image, MODEL, transform=get_transforms(train=False), device=DEVICE)
    return (
        prediction["label"],
        f"{prediction['confidence']*100:.2f}%",
        heatmap,
    )


def build_interface():
    title = "AI Image Detector"
    description = (
        "Upload an image and detect whether it is REAL or AI-generated. "
        "The app returns the predicted label, confidence score, and Grad-CAM heatmap." 
    )
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        with gr.Row():
            image_input = gr.Image(type="pil", label="Input Image")
            output_label = gr.Textbox(label="Predicted Label")
            output_confidence = gr.Textbox(label="Confidence")
        heatmap_output = gr.Image(label="Grad-CAM Heatmap")
        classify_button = gr.Button("Analyze")
        classify_button.click(fn=classify_image, inputs=image_input, outputs=[output_label, output_confidence, heatmap_output])
    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
