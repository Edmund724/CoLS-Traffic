"""
Export trained NASYOLOv8 detector to ONNX format.

Usage:
    # Modify Config below, then run directly:
    python export_onnx_yolo.py

    # Or override via command line:
    python export_onnx_yolo.py --weights runs/bdd100k_nas_yolo/nas_yolo_baseline/best.pt
    python export_onnx_yolo.py --weights runs/coco128_nas_yolo/nas_yolo_test/best.pt --imgsz 320
    python export_onnx_yolo.py --simplify       # requires onnxsim
    # FP16/INT8 quantization should be done at TensorRT stage, not ONNX
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from nas_yolo import NASYOLOv8

# ── Config ────────────────────────────────────────────────────────────────
WEIGHTS   = "runs/bdd100k_nas_yolo/nas_yolo_baseline/best.pt"
IMGSZ     = 320
OPSET     = 11
SIMPLIFY  = False      # requires: pip install onnxsim
FP16      = False       # save weights as fp16 (smaller file, for TensorRT)
OUT_DIR   = "runs/onnx"
# ─────────────────────────────────────────────────────────────────────────


class NASYOLOv8Export(nn.Module):
    """Wraps NASYOLOv8 for ONNX export — forces eval-mode output format."""

    def __init__(self, model: NASYOLOv8):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        preds = self.model(x)
        # eval mode returns (pred_tensor, raw_list)
        # pred_tensor: (batch, 4+nc, num_anchors) — keep only this
        if isinstance(preds, (list, tuple)):
            return preds[0]
        return preds


def export(weights, imgsz, opset, simplify, fp16, out_dir):
    # ── Load model ────────────────────────────────────────────────────────
    model = NASYOLOv8.load(weights)
    model.eval()
    nc = model.nc
    c3, c4, c5 = model.backbone.out_channels

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {weights}")
    print(f"  nc={nc}  params={n_params:.2f}M  input={imgsz}x{imgsz}")
    print(f"  backbone channels: C3={c3} C4={c4} C5={c5}")

    # ── Wrap for export ───────────────────────────────────────────────────
    export_model = NASYOLOv8Export(model)
    export_model.eval()

    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        test_out = export_model(dummy)
    print(f"  output shape: {tuple(test_out.shape)}")

    # ── Export ONNX ───────────────────────────────────────────────────────
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stem = Path(weights).stem
    onnx_path = out_path / f"{stem}_{imgsz}.onnx"

    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},
        },
    )
    print(f"\nONNX exported → {onnx_path}")

    # ── Verify ────────────────────────────────────────────────────────────
    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX check: OK")

    # ── Simplify ──────────────────────────────────────────────────────────
    if simplify:
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, str(onnx_path))
                print(f"  Simplified: OK")
            else:
                print(f"  Simplified: FAILED (keeping original)")
        except ImportError:
            print(f"  [WARN] onnxsim not installed. pip install onnxsim")

    # ── FP16 ──────────────────────────────────────────────────────────────
    if fp16:
        try:
            from onnxconverter_common import float16
            fp16_model = float16.convert_float_to_float16(onnx_model)
            fp16_path = out_path / f"{stem}_{imgsz}_fp16.onnx"
            onnx.save(fp16_model, str(fp16_path))
            print(f"  FP16 exported → {fp16_path}")
        except ImportError:
            print(f"  [WARN] onnxconverter-common not installed. pip install onnxconverter-common")

    # ── File size ─────────────────────────────────────────────────────────
    size_mb = onnx_path.stat().st_size / 1e6
    print(f"\n  File size: {size_mb:.2f} MB")
    print(f"  Input:  images  (batch, 3, {imgsz}, {imgsz})")
    print(f"  Output: output  (batch, {4+nc}, {test_out.shape[2]})")
    print(f"\nDone.")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--weights",   default=WEIGHTS)
    pa.add_argument("--imgsz",     type=int, default=IMGSZ)
    pa.add_argument("--opset",     type=int, default=OPSET)
    pa.add_argument("--simplify",  action="store_true", default=SIMPLIFY)
    pa.add_argument("--fp16",      action="store_true", default=FP16)
    pa.add_argument("--out_dir",   default=OUT_DIR)
    args = pa.parse_args()

    export(args.weights, args.imgsz, args.opset, args.simplify, args.fp16, args.out_dir)
