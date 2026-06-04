"""
NAS Backbone + YOLOv8 Neck + Detection Head

Architecture:
    Backbone  : NASBackboneFeat  → [C3, C4, C5]
    Projection: 1x1 conv to fixed P3/P4/P5 widths
    Neck      : SPPF + PAN-FPN (C2f blocks)
    Head      : Ultralytics Detect
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from backbones import build_backbone, build_backbone_from_spec
from ultralytics.nn.modules.block import SPPF, C2f
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


class NASYOLOv8(nn.Module):
    """
    Args:
        backbone    : backbone module that returns [C3, C4, C5]
        nc          : number of detection classes
        p3_ch       : projected P3 width
        p4_ch       : projected P4 width
        p5_ch       : projected P5 width
    """

    def __init__(
        self,
        backbone: NASBackboneFeat,
        nc: int = 80,
        p3_ch: int = 96,
        p4_ch: int = 192,
        p5_ch: int = 192,
    ):
        super().__init__()
        self.backbone = backbone
        c3_ch, c4_ch, c5_ch = backbone.out_channels

        p3 = int(p3_ch)
        p4 = int(p4_ch)
        p5 = int(p5_ch)
        self.neck_channels = (p3, p4, p5)

        # Project different backbone widths into a detector-stable neck width.
        self.proj_c3 = Conv(c3_ch, p3, k=1, s=1)
        self.proj_c4 = Conv(c4_ch, p4, k=1, s=1)
        self.proj_c5 = Conv(c5_ch, p5, k=1, s=1)

        # ── Top-down FPN ──────────────────────────────────────────────────
        self.sppf = SPPF(p5, p5, k=5)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_p4 = C2f(p5 + p4, p4)  # fuse P5↑ + C4

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c2f_p3 = C2f(p4 + p3, p3)  # fuse P4↑ + C3

        # ── Bottom-up PAN ─────────────────────────────────────────────────
        self.down1 = Conv(p3, p3, k=3, s=2)
        self.c2f_n4 = C2f(p3 + p4, p4)  # fuse P3↓ + P4

        self.down2 = Conv(p4, p4, k=3, s=2)
        self.c2f_n5 = C2f(p4 + p5, p5)  # fuse N4↓ + P5

        # ── Detection head ────────────────────────────────────────────────
        self.detect = Detect(nc=nc, ch=[p3, p4, p5])

        # Let Detect know the input image stride for each feature level
        self.detect.stride = torch.tensor([8.0, 16.0, 32.0])
        self.stride = self.detect.stride
        self.nc = nc

        # Required by v8DetectionLoss: model.args and model.model[-1]
        self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.model = [self.detect]  # loss looks up model.model[-1] for Detect head

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x, return_features=False):
        c3, c4, c5 = self.backbone(x)
        c3_proj = self.proj_c3(c3)
        c4_proj = self.proj_c4(c4)
        c5_proj = self.proj_c5(c5)

        # Top-down
        p5 = self.sppf(c5_proj)
        p4 = self.c2f_p4(torch.cat([self.up1(p5), c4_proj], dim=1))
        p3 = self.c2f_p3(torch.cat([self.up2(p4), c3_proj], dim=1))

        # Bottom-up
        n4 = self.c2f_n4(torch.cat([self.down1(p3), p4], dim=1))
        n5 = self.c2f_n5(torch.cat([self.down2(n4), p5], dim=1))

        detect_out = self.detect([p3, n4, n5])
        if not return_features:
            return detect_out

        features = {
            "c3": c3,
            "c4": c4,
            "c5": c5,
            "c3_proj": c3_proj,
            "c4_proj": c4_proj,
            "c5_proj": c5_proj,
            "p3": p3,
            "p4": p4,
            "p5": p5,
            "n4": n4,
            "n5": n5,
        }
        return detect_out, features

    # ── Convenience builders ──────────────────────────────────────────────
    @classmethod
    def from_checkpoint(
        cls,
        backbone_ckpt: str,
        nc: int = 80,
        p3_ch: int = 96,
        p4_ch: int = 192,
        p5_ch: int = 192,
    ):
        """Build model from NAS backbone checkpoint path."""
        return cls.from_backbone(
            backbone_type="nas",
            backbone_ckpt=backbone_ckpt,
            nc=nc,
            p3_ch=p3_ch,
            p4_ch=p4_ch,
            p5_ch=p5_ch,
        )

    @classmethod
    def from_backbone(
        cls,
        backbone_type: str,
        nc: int = 80,
        p3_ch: int = 96,
        p4_ch: int = 192,
        p5_ch: int = 192,
        backbone_ckpt: str | None = None,
        backbone_pretrained: bool = True,
        pretrained_dir: str | None = None,
    ):
        backbone, spec = build_backbone(
            backbone_type=backbone_type,
            backbone_ckpt=backbone_ckpt,
            backbone_pretrained=backbone_pretrained,
            pretrained_dir=pretrained_dir,
        )
        model = cls(backbone, nc=nc, p3_ch=p3_ch, p4_ch=p4_ch, p5_ch=p5_ch)
        model._backbone_spec = spec
        return model

    @staticmethod
    def _infer_backbone_spec(state_dict: dict) -> dict:
        """从 detector state_dict 的键名推断 NAS backbone 结构。"""
        import re

        # 推断各 stage 的 block 数量
        max_block = {}
        for k in state_dict.keys():
            if k.startswith("backbone.stage"):
                m = re.match(r"backbone\.stage(\d+)\.(\d+)\.", k)
                if m:
                    stage = int(m.group(1))
                    block = int(m.group(2))
                    max_block[stage] = max(max_block.get(stage, -1), block)
        depths = [max_block.get(i, 0) + 1 for i in range(len(max_block))]
        # 推断 stage_widths（从每个 stage 最后一个 block 的 proj conv 输出通道）
        stage_widths = []
        for stage in range(len(max_block)):
            last_block = max_block.get(stage, 0)
            key = f"backbone.stage{stage}.{last_block}.conv.6.weight"
            if key in state_dict:
                stage_widths.append(state_dict[key].shape[0])
            else:
                for k in state_dict.keys():
                    if k.startswith(f"backbone.stage{stage}.") and "conv.6.weight" in k:
                        stage_widths.append(state_dict[k].shape[0])
                        break
        first_ch = state_dict.get("backbone.first_conv.0.weight", torch.zeros(0)).shape[
            0
        ]
        stage_widths = [first_ch] + stage_widths
        return {
            "type": "nas",
            "arch": {
                "config": {
                    "kernel_size": 3,
                    "expand_ratio": 6,
                    "width_multiplier": 1.0,
                    "depths": depths,
                },
                "model_build": {
                    "input_channel": first_ch,
                    "stage_widths": stage_widths,
                    "last_channel": 256,
                },
                "use_dla_head": False,
                "image_size": 32,
            },
        }

    @classmethod
    def load(cls, detector_ckpt: str):
        """Load a self-contained detector checkpoint (no backbone file needed)."""
        ck = torch.load(detector_ckpt, map_location="cpu", weights_only=False)
        backbone_spec = ck.get("backbone_spec")
        if backbone_spec is None and "backbone_arch" in ck:
            backbone_spec = {"type": "nas", "arch": ck["backbone_arch"]}
        if backbone_spec is None:
            print(
                "[WARN] backbone_spec not found in checkpoint, inferring from state_dict keys..."
            )
            backbone_spec = cls._infer_backbone_spec(ck["model_state"])
            print(f"  Inferred backbone_spec: {backbone_spec}")
        backbone = build_backbone_from_spec(backbone_spec)
        neck_channels = ck.get("neck_channels", [128, 256, 256])
        model = cls(
            backbone,
            nc=ck["nc"],
            p3_ch=neck_channels[0],
            p4_ch=neck_channels[1],
            p5_ch=neck_channels[2],
        )
        model.load_state_dict(ck["model_state"])
        print(f"Loaded detector from {detector_ckpt}  nc={ck['nc']}")
        return model

    def save(self, path: str, backbone_ckpt: str = None):
        payload = {
            "model_state": self.state_dict(),
            "nc": self.nc,
            "stride": self.stride,
            "backbone_spec": getattr(self, "_backbone_spec", None),
            "neck_channels": list(self.neck_channels),
        }
        torch.save(payload, path)
        print(f"Saved → {path}")


# ── Sanity check ─────────────────────────────────────────────────────────
# Modify these defaults or override via command line
CKPT = "../results/best_model_test0.90.pth"
NC = 10

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=CKPT, help="NAS backbone checkpoint")
    p.add_argument("--nc", type=int, default=NC, help="num classes")
    args = p.parse_args()

    model = NASYOLOv8.from_checkpoint(args.ckpt, nc=args.nc)
    model.eval()

    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(dummy)

    total = sum(p.numel() for p in model.parameters()) / 1e6
    backbone_p = sum(p.numel() for p in model.backbone.parameters()) / 1e6

    print(f"Total params   : {total:.2f}M")
    print(f"Backbone params: {backbone_p:.2f}M")
    # Detect head returns (pred_tensor, raw_list) in eval mode
    preds = out[0] if isinstance(out, (list, tuple)) else out
    print(f"  predictions: {tuple(preds.shape)}")
    print(
        f"  -> {preds.shape[1] - 4} classes, {preds.shape[2]} anchor points across 3 scales"
    )
