from pathlib import Path
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _normalize_array(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 0)
    if arr.max() == 0:
        return arr
    return arr / arr.max()


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: str = "backbone.conv_head"):
        self.model = model.eval()
        self.target_layer = self._find_target_layer(target_layer)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_target_layer(self, target_layer):
        if isinstance(target_layer, str):
            module = dict(self.model.named_modules()).get(target_layer)
            if module is None:
                raise ValueError(f"Target layer '{target_layer}' not found in model")
            return module
        return target_layer

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_index: int = None) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_index is None:
            target_index = int(output.softmax(dim=1)[0].argmax())
        loss = output[0, target_index]
        loss.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0).cpu().numpy()
        return _normalize_array(cam)


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)


from src.dataset import get_transforms


def generate_gradcam(
    image: Image.Image,
    model: torch.nn.Module,
    transform=None,
    device: str = "cpu",
) -> Image.Image:
    if transform is None:
        transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)
    cam = GradCAM(model).generate_cam(tensor)
    return overlay_cam(image, cam)
