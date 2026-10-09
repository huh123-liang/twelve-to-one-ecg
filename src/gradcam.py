"""1D Grad-CAM (Selvaraju et al., 2017; Lecture 8) for CNNs and Transformers.

The class-specific saliency is computed on `model.cam_layer`. `model.cam_reshape` turns that
layer's output into (B, K, L) - for CNNs it is already (B, channels, time); for ECG-ViT the
token outputs (B, 1+N, D) are reshaped to (B, D, N) after dropping CLS, which is the
"Grad-CAM for Transformer" recipe from Lecture 8 (a reshape_transform).
The map is ReLU'd, upsampled to the input length and normalised to [0, 1].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM1d:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        self.reshape = getattr(model, "cam_reshape", lambda a: a)
        self.h = model.cam_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.acts = output
        if output.requires_grad:                 # forward passes under no_grad (e.g. attention plots) skip this
            output.register_hook(self._save_grad)   # capture d(score)/d(output) during backward

    def _save_grad(self, grad):
        self.grads = grad

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """x: (1, C, T). Returns saliency of length T."""
        self.model.eval()
        with torch.backends.cudnn.flags(enabled=False):   # cuDNN RNN backward needs train mode otherwise
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            self.model.zero_grad()
            logits[0, class_idx].backward()
        acts = self.reshape(self.acts.detach())
        grads = self.reshape(self.grads.detach())
        w = grads.mean(dim=-1, keepdim=True)                   # (1, K, 1) channel importance
        cam = F.relu((w * acts).sum(1, keepdim=True))          # (1, 1, L)
        cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def remove(self):
        self.h.remove()
