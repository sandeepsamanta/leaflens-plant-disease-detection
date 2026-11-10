"""MobileNetV2 split into a frozen stem (features[:14]) and a trainable tail (features[14:] + classifier).

The stem is the expensive part on a CPU, so we run it once per image and cache its output (96 x 14 x 14).
Fine-tuning then only touches the last five inverted-residual / conv blocks and the classifier.
"""
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

SPLIT_AT = 14
FEATURE_SHAPE = (96, 14, 14)


def load_pretrained():
    return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)


class Stem(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.body = base.features[:SPLIT_AT]

    def forward(self, x):
        return self.body(x)


class Tail(nn.Module):
    """features[14:] + global average pool + dropout + linear. `forward_cam` also returns the 7x7 feature map."""

    def __init__(self, base: nn.Module, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.body = base.features[SPLIT_AT:]
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1280, num_classes)

    def forward_maps(self, x):
        return self.body(x)

    def head(self, fmap):
        return self.fc(self.dropout(fmap.mean((2, 3))))

    def forward(self, x):
        return self.head(self.body(x))


class FullModel(nn.Module):
    """Stem + tail, used for export. Returns (logits, 1280x7x7 feature map) so the API can build a CAM."""

    def __init__(self, stem: Stem, tail: Tail):
        super().__init__()
        self.stem, self.tail = stem, tail

    def forward(self, x):
        fmap = self.tail.forward_maps(self.stem(x))
        return self.tail.fc(fmap.mean((2, 3))), fmap
