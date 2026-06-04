"""
Backbone factory for detection models.

Supports:
    - NAS backbone checkpoints from this repo
    - torchvision MobileNetV3-Large with ImageNet pretrained weights
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from nas_backbone_feat import NASBackboneFeat

DEFAULT_PRETRAINED_DIR = "pretrained"
MOBILENETV3_IMAGENET = "mobilenetv3_imagenet"
NAS_BACKBONE = "nas"
BACKBONE_CHOICES = [NAS_BACKBONE, MOBILENETV3_IMAGENET]


def _resolve_pretrained_dir(pretrained_dir: str | None) -> Path:
    base = Path(pretrained_dir) if pretrained_dir else Path.cwd() / DEFAULT_PRETRAINED_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _load_state_dict_file(weights_path: str) -> dict:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            return state["state_dict"]
        if "model_state" in state:
            return state["model_state"]
    return state


class MobileNetV3ImageNetFeat(nn.Module):
    """Return MobileNetV3-Large multi-scale features [C3, C4, C5] at strides [8, 16, 32]."""

    def __init__(
        self,
        weights_path: str | None = None,
        pretrained: bool = True,
        pretrained_dir: str | None = None,
    ):
        super().__init__()
        self.c3_idx = 6
        self.c4_idx = 12
        self.c5_idx = 16
        self._c3_ch = 40
        self._c4_ch = 112
        self._c5_ch = 960

        model = mobilenet_v3_large(weights=None)

        if weights_path:
            state_dict = _load_state_dict_file(weights_path)
            model.load_state_dict(state_dict, strict=True)
        elif pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
            target_dir = _resolve_pretrained_dir(pretrained_dir)
            try:
                state_dict = weights.get_state_dict(
                    progress=True,
                    check_hash=True,
                    model_dir=str(target_dir),
                    file_name=Path(weights.url).name,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch MobileNetV3 ImageNet weights into {target_dir}. "
                    "Provide a local --backbone path or enable network access."
                ) from exc
            model.load_state_dict(state_dict, strict=True)

        self.features = model.features
        self._backbone_spec = {
            "type": MOBILENETV3_IMAGENET,
            "variant": "large",
            "weights_path": str(Path(weights_path).resolve()) if weights_path else None,
            "pretrained": bool(pretrained),
        }

    @classmethod
    def from_spec(cls, spec: dict):
        return cls(weights_path=None, pretrained=False)

    @property
    def out_channels(self):
        return [self._c3_ch, self._c4_ch, self._c5_ch]

    def forward(self, x):
        c3 = c4 = c5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.c3_idx:
                c3 = x
            elif i == self.c4_idx:
                c4 = x
            elif i == self.c5_idx:
                c5 = x
        return [c3, c4, c5]


def build_nas_backbone_spec(backbone_ckpt: str) -> dict:
    ck = torch.load(backbone_ckpt, map_location="cpu")
    return {
        "type": NAS_BACKBONE,
        "arch": {
            "config": ck["config"],
            "model_build": ck.get("model_build", {}),
            "use_dla_head": ck.get("use_dla_head", False),
            "image_size": ck.get("image_size", 32),
        },
    }


def build_backbone(
    backbone_type: str,
    backbone_ckpt: Optional[str] = None,
    backbone_pretrained: bool = True,
    pretrained_dir: str | None = None,
):
    if backbone_type == NAS_BACKBONE:
        if not backbone_ckpt:
            raise ValueError("NAS backbone requires a checkpoint path")
        backbone = NASBackboneFeat(backbone_ckpt)
        spec = build_nas_backbone_spec(backbone_ckpt)
        return backbone, spec

    if backbone_type == MOBILENETV3_IMAGENET:
        weights_path = None
        if backbone_ckpt:
            ckpt_path = Path(backbone_ckpt)
            if ckpt_path.exists():
                weights_path = str(ckpt_path)
            elif ckpt_path.suffix in {".pt", ".pth"}:
                raise FileNotFoundError(f"MobileNetV3 weights file not found: {ckpt_path}")
        backbone = MobileNetV3ImageNetFeat(
            weights_path=weights_path,
            pretrained=backbone_pretrained and weights_path is None,
            pretrained_dir=pretrained_dir,
        )
        return backbone, backbone._backbone_spec

    raise ValueError(f"Unsupported backbone type: {backbone_type}")


def build_backbone_from_spec(spec: dict):
    backbone_type = spec.get("type", NAS_BACKBONE)
    if backbone_type == NAS_BACKBONE:
        return NASBackboneFeat.from_arch(spec["arch"])
    if backbone_type == MOBILENETV3_IMAGENET:
        return MobileNetV3ImageNetFeat.from_spec(spec)
    raise ValueError(f"Unsupported backbone spec type: {backbone_type}")


def describe_backbone(spec: dict | None) -> str:
    if not spec:
        return "Unknown backbone"
    if spec.get("type") == NAS_BACKBONE:
        return "NAS-searched MobileNetV3"
    if spec.get("type") == MOBILENETV3_IMAGENET:
        return "MobileNetV3-Large ImageNet"
    return spec.get("type", "Unknown backbone")
