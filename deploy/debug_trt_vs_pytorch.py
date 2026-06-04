#!/usr/bin/env python3
"""
Debug script: compare PyTorch vs TensorRT outputs on the SAME batch.
Compatible with TensorRT 8.5+ / 10.x API.

Usage:
    python debug_trt_vs_pytorch.py \
        --engine qat_runs/qat_dair_v1/qat_int8.engine \
        --weights distillation/runs/distill_dair/distill_d1_at/best_ema.pt \
        --batch-idx 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

_HERE = Path(__file__).resolve().parent
_DETECTION_DIR = _HERE / "detection"
sys.path.insert(0, str(_DETECTION_DIR))

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset

from nas_yolo import NASYOLOv8

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    _HAS_TRT = True
except Exception as e:
    _HAS_TRT = False
    print(f"[ERROR] TensorRT / pycuda not available: {e}")
    sys.exit(1)


class TRTInference:
    """TensorRT inference with CUDA context management (coexists with PyTorch)."""

    def __init__(self, engine_path: str):
        # Create isolated CUDA context to avoid conflict with PyTorch
        self.cuda_ctx = cuda.Device(0).make_context()

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            d = {"name": name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(d)
            else:
                self.outputs.append(d)

            self.context.set_tensor_address(name, int(device_mem))

        self.cuda_ctx.pop()
        print(f"[TRT] Loaded engine: {engine_path}")
        print(f"[TRT] Input shape : {self.inputs[0]['shape']}")
        print(f"[TRT] Output shape: {self.outputs[0]['shape']}")

    def infer(self, input_tensor: np.ndarray):
        self.cuda_ctx.push()
        try:
            np.copyto(self.inputs[0]["host"], input_tensor.ravel())
            cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
            self.context.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
            self.stream.synchronize()
            return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        finally:
            self.cuda_ctx.pop()


def get_val_loader(data_yaml: str, imgsz: int = 640, batch: int = 1, workers: int = 0):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data_yaml
    cfg.imgsz = imgsz
    cfg.batch = batch
    cfg.workers = workers
    cfg.task = "detect"
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0
    cfg.scale = 0.5
    cfg.mosaic = 1.0
    data_info = check_det_dataset(cfg.data)
    val_ds = build_yolo_dataset(cfg, data_info["val"], batch, data_info, mode="val", rect=False)
    return build_dataloader(val_ds, batch, workers, shuffle=False, rank=-1)


def postprocess(pred_tensor, conf=0.25, iou=0.45, img_h=640, img_w=640):
    bs = pred_tensor.shape[0]
    all_results = []

    if pred_tensor.shape[1] == 14 and pred_tensor.shape[2] == 8400:
        for i in range(bs):
            p = pred_tensor[i]
            boxes_xywh = p[:4]
            cls_scores = p[4:]
            all_results.append((boxes_xywh, cls_scores))
    elif pred_tensor.shape[1] == 8400 and pred_tensor.shape[2] == 14:
        for i in range(bs):
            p = pred_tensor[i]
            boxes_xywh = p[:, :4].T
            cls_scores = p[:, 4:].T
            all_results.append((boxes_xywh, cls_scores))
    else:
        raise ValueError(f"Unexpected pred shape: {pred_tensor.shape}")

    preds_list = []
    for boxes_xywh, cls_scores in all_results:
        x1 = boxes_xywh[0] - boxes_xywh[2] / 2
        y1 = boxes_xywh[1] - boxes_xywh[3] / 2
        x2 = boxes_xywh[0] + boxes_xywh[2] / 2
        y2 = boxes_xywh[1] + boxes_xywh[3] / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
        boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, img_w)
        boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, img_h)

        scores, labels = cls_scores.max(dim=0)
        mask = scores > conf
        boxes_xyxy = boxes_xyxy[mask]
        scores = scores[mask]
        labels = labels[mask]

        if len(scores):
            max_wh = max(img_h, img_w) + 1
            nms_boxes = boxes_xyxy + labels.unsqueeze(1).to(boxes_xyxy.dtype) * max_wh
            keep = torchvision.ops.nms(nms_boxes, scores, iou)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            labels = labels[keep]

        preds_list.append({"boxes": boxes_xyxy, "scores": scores, "labels": labels})
    return preds_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=str(_HERE / "datasets" / "dair_v2x_yolo" / "dair_v2x.yaml"))
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_loader = get_val_loader(args.data, args.imgsz, args.batch, workers=0)

    batch_data = None
    for idx, bd in enumerate(val_loader):
        if idx == args.batch_idx:
            batch_data = bd
            break
    if batch_data is None:
        print(f"[ERROR] batch_idx {args.batch_idx} out of range")
        return

    imgs = batch_data["img"].to(device).float() / 255.0
    print(f"[Info] Batch {args.batch_idx} | img shape: {imgs.shape}")

    # PyTorch
    print("\n" + "=" * 60)
    print("PyTorch FP32 Inference")
    print("=" * 60)
    model = NASYOLOv8.load(args.weights).to(device).eval()
    with torch.no_grad():
        pt_out = model(imgs)
        pt_tensor = pt_out[0] if isinstance(pt_out, (list, tuple)) else pt_out
    print(f"PyTorch output shape: {pt_tensor.shape}")
    print(f"PyTorch output range: [{pt_tensor.min():.4f}, {pt_tensor.max():.4f}]")
    print(f"PyTorch output mean: {pt_tensor.mean():.4f}")

    pt_preds = postprocess(pt_tensor, conf=0.25, iou=0.45)
    print(f"PyTorch detections: {len(pt_preds[0]['boxes'])} boxes")
    if len(pt_preds[0]['boxes']) > 0:
        print(f"  Top-5 scores: {pt_preds[0]['scores'][:5].cpu().numpy()}")
        print(f"  Top-5 labels: {pt_preds[0]['labels'][:5].cpu().numpy()}")

    # TensorRT
    print("\n" + "=" * 60)
    print("TensorRT Inference")
    print("=" * 60)
    trt_infer = TRTInference(args.engine)
    input_np = imgs.cpu().numpy().astype(np.float32)
    trt_out_np = trt_infer.infer(input_np)
    print(f"TRT output shape: {trt_out_np.shape}")

    trt_tensor = torch.from_numpy(trt_out_np).to(device)
    print(f"TRT output range: [{trt_tensor.min():.4f}, {trt_tensor.max():.4f}]")
    print(f"TRT output mean: {trt_tensor.mean():.4f}")

    # Numerical comparison
    print("\n" + "=" * 60)
    print("Numerical Comparison (PyTorch vs TRT)")
    print("=" * 60)

    layouts = [
        ("(batch, 14, 8400)", lambda t: t),
        ("(batch, 8400, 14)", lambda t: t.transpose(1, 2)),
    ]
    best_diff = float("inf")
    best_layout = None

    for name, transform in layouts:
        try:
            t_trt = transform(trt_tensor)
            if t_trt.shape != pt_tensor.shape:
                continue
            diff = (pt_tensor - t_trt).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f"  Layout {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            if max_diff < best_diff:
                best_diff = max_diff
                best_layout = name
                best_trt_tensor = t_trt.clone()
        except Exception as e:
            print(f"  Layout {name}: failed ({e})")

    if best_layout is None:
        print("[ERROR] No compatible layout found!")
        return

    print(f"\n  Best match: {best_layout} | max_diff={best_diff:.6f}")
    diff_tensor = (pt_tensor - best_trt_tensor).abs()
    for c in range(14):
        c_diff = diff_tensor[0, c].max().item() if diff_tensor.dim() == 3 else diff_tensor[0, :, c].max().item()
        print(f"    Channel {c:2d} max_diff: {c_diff:.6f}")

    # Post-processing comparison
    print("\n" + "=" * 60)
    print("Post-processing Comparison")
    print("=" * 60)
    trt_preds = postprocess(best_trt_tensor, conf=0.25, iou=0.45)
    print(f"TRT detections  : {len(trt_preds[0]['boxes'])} boxes")
    if len(trt_preds[0]['boxes']) > 0:
        print(f"  Top-5 scores: {trt_preds[0]['scores'][:5].cpu().numpy()}")
        print(f"  Top-5 labels: {trt_preds[0]['labels'][:5].cpu().numpy()}")

    n_pt = len(pt_preds[0]['boxes'])
    n_trt = len(trt_preds[0]['boxes'])
    if n_pt == n_trt and n_pt > 0:
        box_diff = (pt_preds[0]['boxes'] - trt_preds[0]['boxes']).abs()
        score_diff = (pt_preds[0]['scores'] - trt_preds[0]['scores']).abs()
        print(f"\n  Box diff   : max={box_diff.max():.4f}, mean={box_diff.mean():.4f}")
        print(f"  Score diff : max={score_diff.max():.6f}, mean={score_diff.mean():.6f}")
    else:
        print(f"\n  Box count mismatch: PyTorch={n_pt}, TRT={n_trt}")

    print("\n" + "=" * 60)
    print("Diagnosis")
    print("=" * 60)
    if best_diff < 1e-4:
        print("  -> TRT output NUMERICALLY matches PyTorch. mAP drop is likely in POST-PROCESSING.")
    elif best_diff < 0.1:
        print("  -> TRT output has MINOR numeric diff (fp rounding). mAP drop should be <0.5%.")
    else:
        print("  -> TRT output has SIGNIFICANT numeric diff. Check INT8 calibration / engine build.")
    print("=" * 60)


if __name__ == "__main__":
    main()
