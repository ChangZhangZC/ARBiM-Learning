"""Native ImageNet ResNet18 with spatial pooling and frozen weights."""
import torch
from torch import nn
from .depth_preprocess import preprocess

class DepthEncoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2],
                                     nn.AdaptiveAvgPool2d((2,2)), nn.Flatten(1))
        self.to(device).requires_grad_(False).eval()
    @torch.inference_mode()
    def forward(self, depth, roi=None):
        x = preprocess(depth, roi, next(self.parameters()).device)
        return self.encoder(x).cpu().numpy()
