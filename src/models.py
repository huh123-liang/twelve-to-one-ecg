"""Model zoo. Every model takes (B, C, T) and returns logits (B, n_classes).

Three architecture families (one inductive bias each, compared with one training loop - Lecture 5):
* ResNet1d     - convolution + residual identity blocks         (Lectures 2, 3, 8)
* BiLSTMAttn   - CNN-BiLSTM hybrid with attention pooling       (Lectures 4, 5 "build the hybrid")
* ECGViT       - patch embedding + hand-written multi-head self-attention, CLS token,
                 sinusoidal positional encoding                  (Lectures 6, 7)
Our design:
* LA-ResNet    - ResNet1d that receives 12 signal channels + 12 lead-mask channels and is
                 trained with random lead dropping, so one model serves any lead subset.

Ablation switches (Lecture exercises): residual on/off (L3 VGG vs L8 ResNet), kernel size /
receptive field (L5 exercise 1), LSTM vs vanilla RNN (L4).

Each model exposes `cam_layer` (module whose output Grad-CAM uses) and `cam_reshape`
(turns that output into (B, K, L)), so Grad-CAM works for CNNs and Transformers (Lecture 8).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_reshape(a):
    return a


# ----------------------------------------------------------------------------
# ResNet1d
# ----------------------------------------------------------------------------

class BasicBlock1d(nn.Module):
    """Two conv layers + identity shortcut (Lecture 8: ResNet identity block)."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, k: int = 7, drop: float = 0.0,
                 residual: bool = True):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, k, stride=stride, padding=k // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)
        self.residual = residual
        self.short = None
        if residual and (stride != 1 or c_in != c_out):
            self.short = nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if self.residual:                       # x + F(x): the line that makes deep CNNs trainable
            out = out + (x if self.short is None else self.short(x))
        return self.act(out)


class ResNet1d(nn.Module):
    def __init__(self, n_leads: int, n_classes: int = 5, width: int = 64, drop: float = 0.2,
                 kernel: int = 7, residual: bool = True):
        super().__init__()
        w, k, r = width, kernel, residual
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, w, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(w), nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1))
        self.layers = nn.Sequential(
            BasicBlock1d(w, w, k=k, residual=r), BasicBlock1d(w, w, k=k, residual=r),
            BasicBlock1d(w, 2 * w, stride=2, k=k, residual=r), BasicBlock1d(2 * w, 2 * w, k=k, residual=r),
            BasicBlock1d(2 * w, 4 * w, stride=2, k=k, residual=r), BasicBlock1d(4 * w, 4 * w, k=k, residual=r),
            BasicBlock1d(4 * w, 4 * w, stride=2, k=k, drop=drop, residual=r),
            BasicBlock1d(4 * w, 4 * w, k=k, residual=r))
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(8 * w, n_classes))
        self.cam_reshape = _identity_reshape

    @property
    def cam_layer(self):          # a property (not an attribute) so it is not saved twice in state_dict
        return self.layers[-1]

    def receptive_field(self) -> int:
        """Receptive field in input samples (100 Hz -> divide by 100 for seconds)."""
        rf, jump = 1, 1
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.MaxPool1d)):
                ks = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
                st = m.stride[0] if isinstance(m.stride, tuple) else m.stride
                if ks == 1:
                    continue  # 1x1 shortcut convs do not enlarge the field
                rf += (ks - 1) * jump
                jump *= st
        return rf

    def forward(self, x):
        h = self.layers(self.stem(x))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)  # average + max pooling
        return self.head(h)


def LAResNet(n_leads: int = 24, n_classes: int = 5, **kw):
    """Lead-agnostic ResNet: 12 signal channels + 12 mask channels, same backbone as ResNet1d."""
    return ResNet1d(n_leads=24, n_classes=n_classes, **kw)


# ----------------------------------------------------------------------------
# CNN-BiLSTM with attention pooling
# ----------------------------------------------------------------------------

class ConvStem(nn.Module):
    """Downsample raw ECG by `factor` (power of 2) with Conv1d + BatchNorm blocks (Lecture 5)."""

    def __init__(self, n_leads: int, d: int, factor: int = 4):
        super().__init__()
        n = int(math.log2(factor))
        chans = [n_leads] + [d] * n
        blocks = []
        for i in range(n):
            blocks += [nn.Conv1d(chans[i], chans[i + 1], 7, stride=2, padding=3, bias=False),
                       nn.BatchNorm1d(chans[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*blocks)
        self._last_idx = len(blocks) - 3  # last conv, used for Grad-CAM

    @property
    def last(self):
        return self.net[self._last_idx]

    def forward(self, x):
        return self.net(x)


class BiLSTMAttn(nn.Module):
    """CNN front-end + bidirectional recurrent layers + attention pooling (Lectures 4, 5).

    cell="lstm" (default) or "rnn" (vanilla RNN, for the Lecture 4 vanishing-gradient ablation).
    """

    def __init__(self, n_leads: int, n_classes: int = 5, d: int = 128, hidden: int = 128,
                 layers: int = 2, drop: float = 0.3, cell: str = "lstm"):
        super().__init__()
        self.stem = ConvStem(n_leads, d, factor=4)
        rnn_cls = {"lstm": nn.LSTM, "rnn": nn.RNN}[cell]
        self.rnn = rnn_cls(d, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                           dropout=drop if layers > 1 else 0.0)
        self.attn = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * hidden, n_classes))
        self.cam_reshape = _identity_reshape

    @property
    def cam_layer(self):
        return self.stem.last

    def forward(self, x):
        h = self.stem(x).transpose(1, 2)          # (B, L, d)   batch_first layout (Lecture 4)
        h, _ = self.rnn(h)                        # (B, L, 2H)  output at every step
        a = torch.softmax(self.attn(h), dim=1)    # (B, L, 1)   which time steps matter
        return self.head((a * h).sum(1))


# ----------------------------------------------------------------------------
# ECG-ViT: the Lecture 7 Vision Transformer with 1-D signal patches
# ----------------------------------------------------------------------------

def sinusoidal_position_encoding(n_positions: int, dim: int) -> torch.Tensor:
    """[T, D] sinusoidal encoding (Lecture 7, section 6)."""
    pos = torch.arange(n_positions).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(n_positions, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MultiHeadSelfAttention(nn.Module):
    """Hand-written multi-head self-attention, same structure as Lecture 7.

    Q = what I am looking for, K = what I offer, V = what I pass on (Lecture 6).
    The attention weights of the last forward pass are kept in `self.last_weights`
    so we can plot which ECG segments the CLS token attends to.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if num_heads < 1 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by a positive num_heads.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_layer = nn.Linear(embed_dim, embed_dim)
        self.k_layer = nn.Linear(embed_dim, embed_dim)
        self.v_layer = nn.Linear(embed_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_weights = None

    def forward(self, x):                                   # x: [B, T, D]
        B, T, D = x.shape
        Q = self.q_layer(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, d]
        K = self.k_layer(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_layer(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)                  # [B, h, T, T]
        weights = torch.softmax(scores, dim=-1)              # every row sums to 1
        self.last_weights = weights.detach()
        out = self.dropout(weights) @ V                      # [B, h, T, d] weighted average of values
        out = out.transpose(1, 2).reshape(B, T, D)           # concatenate the heads
        return self.output_layer(out)


class AttentionBlock(nn.Module):
    """Pre-LayerNorm Transformer block: x = x + MHSA(LN(x)); x = x + FFN(LN(x)) (Lectures 6, 7)."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.layer_norm_1(x))   # residual: ADD, do not overwrite
        x = x + self.linear(self.layer_norm_2(x))
        return x


class ECGViT(nn.Module):
    """1-D Vision Transformer for ECG.

    The 10 s recording (T samples) is cut into non-overlapping patches of `patch_size`
    samples (default 20 = 200 ms at 100 Hz), each patch of all leads is flattened and
    linearly embedded - exactly the image-patch recipe of Lecture 7 with 1-D patches.
    """

    def __init__(self, n_leads: int, n_classes: int = 5, seq_len: int = 1000, patch_size: int = 20,
                 embed_dim: int = 128, hidden_dim: int = 512, num_heads: int = 4, num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size")
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.input_layer = nn.Linear(n_leads * patch_size, embed_dim)          # patch embedding W_E
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)     # [1, 1, D]
        self.register_buffer("pos_encoding", sinusoidal_position_encoding(self.num_patches + 1, embed_dim).unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
                                     for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_classes))
        self.cam_reshape = self._tokens_to_cam

    @property
    def cam_layer(self):
        # Grad-CAM must hook the INPUT side of the last block: after the last block only the CLS
        # token reaches the classifier, so patch tokens there receive zero gradient.
        return self.blocks[-1].layer_norm_1

    @staticmethod
    def _tokens_to_cam(a):
        """[B, 1+N, D] token outputs -> [B, D, N] (drop CLS) so Grad-CAM can treat tokens as time steps."""
        return a[:, 1:, :].transpose(1, 2)

    def to_patches(self, x):                        # x: [B, C, T]
        B, C, T = x.shape
        p = x.unfold(2, self.patch_size, self.patch_size)   # [B, C, N, P]
        return p.permute(0, 2, 1, 3).reshape(B, -1, C * self.patch_size)  # [B, N, C*P]

    def forward(self, x):
        z = self.input_layer(self.to_patches(x))                     # [B, N, D]
        cls = self.cls_token.expand(z.size(0), -1, -1)               # one CLS copy per record
        z = torch.cat([cls, z], dim=1) + self.pos_encoding           # [B, 1+N, D]
        z = self.dropout(z)
        for block in self.blocks:
            z = block(z)
        return self.mlp_head(z[:, 0])                                # classify from CLS

    def cls_attention(self) -> torch.Tensor:
        """Attention of CLS to every patch in the last block, averaged over heads: [B, N]."""
        w = self.blocks[-1].attn.last_weights                         # [B, h, 1+N, 1+N]
        return w[:, :, 0, 1:].mean(1)


class MaskedSignalModel(nn.Module):
    """Reconstruct randomly hidden ECG patches without labels (Lecture 7 self-supervision)."""

    def __init__(self, n_leads: int, mask_ratio: float = 0.15):
        super().__init__()
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be between zero and one")
        self.encoder = ECGViT(n_leads=n_leads, n_classes=1)
        self.mask_ratio = mask_ratio
        dim = self.encoder.input_layer.out_features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_leads * self.encoder.patch_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        patches = self.encoder.to_patches(x)
        batch, n_patches, _ = patches.shape
        if mask is None:
            n_hidden = round(self.mask_ratio * n_patches)
            order = torch.rand(batch, n_patches, device=x.device).argsort(dim=1)
            mask = torch.zeros(batch, n_patches, dtype=torch.bool, device=x.device)
            mask.scatter_(1, order[:, :n_hidden], True)
        elif mask.shape != (batch, n_patches) or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean [batch, patches] tensor")
        tokens = self.encoder.input_layer(patches)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        cls = self.encoder.cls_token.expand(batch, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.encoder.pos_encoding
        tokens = self.encoder.dropout(tokens)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        reconstruction = self.decoder(tokens[:, 1:])
        loss = (reconstruction.float() - patches.float()).square()[mask].mean()
        return loss, reconstruction, mask


# ----------------------------------------------------------------------------
# Registry and helpers
# ----------------------------------------------------------------------------

MODELS = {"resnet1d": ResNet1d, "bilstm": BiLSTMAttn, "vit": ECGViT, "laresnet": LAResNet}
MODEL_FAMILY = {"resnet1d": "ResNet1d", "bilstm": "CNN-BiLSTM", "vit": "ECG-ViT", "laresnet": "LA-ResNet"}


def build_model(name: str, n_leads: int, n_classes: int = 5, kernel: int = 7, residual: bool = True,
                cell: str = "lstm") -> nn.Module:
    if name in ("resnet1d", "laresnet"):
        return MODELS[name](n_leads=n_leads, n_classes=n_classes, kernel=kernel, residual=residual)
    if name == "bilstm":
        return BiLSTMAttn(n_leads=n_leads, n_classes=n_classes, cell=cell)
    if name == "vit":
        return ECGViT(n_leads=n_leads, n_classes=n_classes)
    raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")


def load_weights(model: nn.Module, path: str, map_location="cpu") -> nn.Module:
    """Load a checkpoint; drops the duplicated 'cam_layer.*' / 'stem.last.*' keys that v1 checkpoints contain."""
    sd = torch.load(path, map_location=map_location)
    sd = {k: v for k, v in sd.items() if not (k.startswith("cam_layer.") or ".last." in k)}
    model.load_state_dict(sd)
    return model


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def replace_input_layer(model: ResNet1d, new_in: int, keep_channels: list[int] | None = None) -> ResNet1d:
    """Transfer learning (Lecture 8): swap the first conv for one with `new_in` input leads.

    If `keep_channels` lists which of the old input leads are still present, their pretrained
    filters are copied into the new layer, so the reduced-lead model starts from the 12-lead
    filters for those leads instead of from random weights.
    """
    old = model.stem[0]
    new = nn.Conv1d(new_in, old.out_channels, old.kernel_size[0], stride=old.stride[0],
                    padding=old.padding[0], bias=False)
    if keep_channels is not None and len(keep_channels) == new_in:
        with torch.no_grad():
            new.weight.copy_(old.weight[:, keep_channels, :])
    model.stem[0] = new
    return model
