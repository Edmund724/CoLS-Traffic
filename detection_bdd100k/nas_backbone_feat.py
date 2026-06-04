"""
NAS backbone → multi-scale feature extractor for detection.

Loads a pretrained MobileNetV3Cifar checkpoint and splits its flat block list
into 4 stages, then returns [C3, C4, C5] for FPN-based detectors.

Spatial strides (input 640×640):
    first_conv      : /2   → 320×320
    stage 0 (s=2)   : /4   → 160×160
    stage 1 (s=2)   : /8   → 80×80   → C3
    stage 2 (s=2)   : /16  → 40×40   → C4
    stage 3 (s=1)   : /16  → 40×40
    extra_conv(s=2) : /32  → 20×20   → C5
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path so nas_fbnet can be imported
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from nas_fbnet.models.mbconv import make_divisible


class NASBackboneFeat(nn.Module):
    """
    Args:
        checkpoint_path: path to a .pth checkpoint saved by nas_fbnet train.py
                         Supports both MobileNetV3Cifar and MobileNetV3CifarDLAHead.
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        ck = torch.load(checkpoint_path, map_location="cpu")
        arch_config = ck["config"]

        # ── Read actual build params from checkpoint (handles custom stage_widths) ──
        model_build   = ck.get("model_build", {})
        w_mult        = arch_config.get("width_multiplier", 1.0)
        base_widths   = model_build.get("stage_widths", [16, 24, 40, 80, 160])
        input_channel = model_build.get("input_channel", 16)
        last_channel  = model_build.get("last_channel", 256)
        image_size    = model_build.get("cifar_image_size", 32)

        stage_widths  = [make_divisible(w * w_mult) for w in base_widths]

        depths = arch_config["depths"]   # [d0, d1, d2, d3]

        # C3/C4/C5 output channels
        self._c3_ch = stage_widths[2]
        self._c4_ch = stage_widths[3]
        self._c5_ch = stage_widths[4]

        # ── Rebuild the correct model class to load weights ───────────────
        use_dla = ck.get("use_dla_head", False)
        if use_dla:
            from nas_fbnet.models.mobilenet_v3_dla import MobileNetV3CifarDLAHead
            full = MobileNetV3CifarDLAHead(
                arch_config,
                input_channel=input_channel,
                stage_widths=base_widths,
                last_channel=last_channel,
                image_size=image_size,
            )
        else:
            from nas_fbnet.models.mobilenet_v3 import MobileNetV3Cifar
            full = MobileNetV3Cifar(
                arch_config,
                input_channel=input_channel,
                stage_widths=base_widths,
                last_channel=last_channel,
            )

        missing, unexpected = full.load_state_dict(ck["state_dict"], strict=False)
        if missing:
            print(f"[NASBackboneFeat] missing keys: {missing}")

        # ── Carve out backbone layers only (drop classification head) ─────
        self.first_conv = full.first_conv

        all_blocks = list(full.blocks.children())
        d0, d1, d2, d3 = depths
        self.stage0 = nn.Sequential(*all_blocks[:d0])
        self.stage1 = nn.Sequential(*all_blocks[d0: d0+d1])          # → C3
        self.stage2 = nn.Sequential(*all_blocks[d0+d1: d0+d1+d2])    # → C4
        self.stage3 = nn.Sequential(*all_blocks[d0+d1+d2:])

        # Extra stride-2 dw-sep conv: stride 16 → stride 32 (C5)
        c = self._c5_ch
        self.extra_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU6(inplace=True),
        )

    # ── Class method: rebuild from saved arch dict (no .pth file needed) ────
    @classmethod
    def from_arch(cls, arch: dict):
        """Reconstruct backbone structure from arch dict saved inside detector checkpoint.
        Weights are loaded from the detector's model_state, not from a backbone file.
        This creates an empty (randomly initialized) backbone with the correct structure."""
        import tempfile, os
        # We need a dummy checkpoint to reuse __init__; create a minimal one in memory
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)

        arch_config  = arch["config"]
        model_build  = arch.get("model_build", {})
        use_dla      = arch.get("use_dla_head", False)
        image_size   = arch.get("image_size", 32)

        w_mult       = arch_config.get("width_multiplier", 1.0)
        base_widths  = model_build.get("stage_widths", [16, 24, 40, 80, 160])
        input_channel= model_build.get("input_channel", 16)
        last_channel = model_build.get("last_channel", 256)
        stage_widths = [make_divisible(w * w_mult) for w in base_widths]
        depths       = arch_config["depths"]

        obj._c3_ch = stage_widths[2]
        obj._c4_ch = stage_widths[3]
        obj._c5_ch = stage_widths[4]

        if use_dla:
            from nas_fbnet.models.mobilenet_v3_dla import MobileNetV3CifarDLAHead
            full = MobileNetV3CifarDLAHead(arch_config, input_channel=input_channel,
                                            stage_widths=base_widths, last_channel=last_channel,
                                            image_size=image_size)
        else:
            from nas_fbnet.models.mobilenet_v3 import MobileNetV3Cifar
            full = MobileNetV3Cifar(arch_config, input_channel=input_channel,
                                     stage_widths=base_widths, last_channel=last_channel)

        obj.first_conv = full.first_conv
        all_blocks = list(full.blocks.children())
        d0, d1, d2, d3 = depths
        obj.stage0 = nn.Sequential(*all_blocks[:d0])
        obj.stage1 = nn.Sequential(*all_blocks[d0: d0+d1])
        obj.stage2 = nn.Sequential(*all_blocks[d0+d1: d0+d1+d2])
        obj.stage3 = nn.Sequential(*all_blocks[d0+d1+d2:])

        c = obj._c5_ch
        obj.extra_conv = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.ReLU6(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU6(inplace=True),
        )
        return obj

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def out_channels(self):
        """[C3_ch, C4_ch, C5_ch]"""
        return [self._c3_ch, self._c4_ch, self._c5_ch]

    # ── Forward ───────────────────────────────────────────────────────────
    def forward(self, x):
        x = self.first_conv(x)   # /2
        x = self.stage0(x)       # /4
        c3 = self.stage1(x)      # /8
        c4 = self.stage2(c3)     # /16
        c5 = self.extra_conv(self.stage3(c4))  # /32
        return [c3, c4, c5]


# ── Quick sanity check ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", default="../results/best_model_test0.90.pth", help="path to .pth checkpoint")
    args = p.parse_args()

    backbone = NASBackboneFeat(args.ckpt)
    backbone.eval()
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        feats = backbone(dummy)
    print(f"out_channels: {backbone.out_channels}")
    for i, f in enumerate(feats):
        print(f"  C{i+3}: {tuple(f.shape)}  stride={640 // f.shape[-1]}")
