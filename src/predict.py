from pathlib import Path
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import torch
from PIL import Image

from config import CLASS_NAMES, DEVICE
from src.dataset import get_transforms
from src.model import EfficientNetFFTClassifier


def load_image(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise ValueError("Unsupported image type for prediction")
    return image


def build_model(checkpoint_path: Path, device: str = DEVICE) -> EfficientNetFFTClassifier:
    model = EfficientNetFFTClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"] if isinstance(checkpoint, dict) else checkpoint)
    model.to(device).eval()
    return model


def predict_image(image, model, device: str = DEVICE):
    image = load_image(image)
    transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    label_idx = int(probabilities.argmax())
    return {
        "label": CLASS_NAMES[label_idx],
        "confidence": float(probabilities[label_idx]),
        "probabilities": probabilities.tolist(),
    }
