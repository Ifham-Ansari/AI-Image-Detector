from pathlib import Path
import sys

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetFFTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.spatial_dim = self.backbone.num_features

        self.fft_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.spatial_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_features = self.backbone(x)
        fft_input = self._prepare_fft_input(x)
        fft_features = self.fft_branch(fft_input).flatten(1)
        combined = torch.cat((spatial_features, fft_features), dim=1)
        return self.classifier(combined)

    def _prepare_fft_input(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.fft2(x, norm="ortho")
        fft = torch.view_as_real(fft)
        magnitude = torch.sqrt(fft[..., 0] ** 2 + fft[..., 1] ** 2)
        magnitude = torch.log1p(magnitude)
        return magnitude
