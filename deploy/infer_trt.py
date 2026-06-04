#!/usr/bin/env python3
"""
TensorRT Python API 推理 + mAP 验证脚本（Jetson 用，内存优化版）。
分段评估，定期清理显存，避免 Jetson 小内存 kill。

Usage:
    python infer_trt.py \
        --engine qat_int8.engine \
        --data datasets/dair_v2x_yolo/dair_v2x.yaml \
        --batch 1 --conf 0.25 --iou 0.45
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

from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG

# ------------------------------------------------------------------
# TensorRT
# ------------------------------------------------------------------
try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    _HAS_TRT = True
except Exception as e:
    _HAS_TRT = False
    print(f"[ERROR] TensorRT / pycuda not available: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# torchmetrics mAP
# ------------------------------------------------------------------
try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False
    print("[WARN] torchmetrics not available. mAP will not be computed.")


class TRTInference:
    """TensorRT engine wrapper with CUDA buffer management – context‑safe."""

    def __init__(self, engine_path: str):
        # 1. 强制创建并保存一个独立的 CUDA context
        self.cuda_ctx = cuda.Device(0).make_context()
        # 此时该 context 已被 push，成为当前上下文

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # 创建执行上下文和流（都在当前 context 下）
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.inputs = []
        self.outputs = []

        # 分配设备内存（在当前 context 下）
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(
                    {
                        "name": name,
                        "host": host_mem,
                        "device": device_mem,
                        "shape": shape,
                        "dtype": dtype,
                    }
                )
            else:
                self.outputs.append(
                    {
                        "name": name,
                        "host": host_mem,
                        "device": device_mem,
                        "shape": shape,
                        "dtype": dtype,
                    }
                )

        # 推入完成后可以 pop 出栈，后续在 infer 中再 push
        self.cuda_ctx.pop()

        print(f"[TRT] Loaded engine: {engine_path}")
        print(f"[TRT] Inputs : {[inp['name'] for inp in self.inputs]}")
        print(f"[TRT] Outputs: {[out['name'] for out in self.outputs]}")

    def infer(self, input_tensor: np.ndarray):
        """
        每次推理前 push 我们自己的 CUDA context，
        结束后 pop，避免与 PyTorch 的 context 冲突。
        """
        # 将自己的 context 压入栈顶
        self.cuda_ctx.push()

        try:
            # 设置所有 I/O 张量的设备地址
            for inp in self.inputs:
                self.context.set_tensor_address(inp["name"], int(inp["device"]))
            for out in self.outputs:
                self.context.set_tensor_address(out["name"], int(out["device"]))

            # 复制输入
            np.copyto(self.inputs[0]["host"], input_tensor.ravel())
            cuda.memcpy_htod_async(
                self.inputs[0]["device"], self.inputs[0]["host"], self.stream
            )

            # 异步执行
            self.context.execute_async_v3(self.stream.handle)

            # 取回输出（仅第一个输出，因为后处理只用到 outputs[0]）
            cuda.memcpy_dtoh_async(
                self.outputs[0]["host"], self.outputs[0]["device"], self.stream
            )
            self.stream.synchronize()

            return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])

        finally:
            # 恢复之前的 context（通常是 PyTorch 的）
            self.cuda_ctx.pop()


# ---- The rest of the script remains unchanged ----
def get_val_loader(data_yaml: str, imgsz: int = 640, batch: int = 1, workers: int = 0):
    """workers=0 for Jetson to avoid multi-process memory overhead."""
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
    val_ds = build_yolo_dataset(
        cfg, data_info["val"], batch, data_info, mode="val", rect=False
    )
    return build_dataloader(val_ds, batch, workers, shuffle=False, rank=-1)


def to_cpu_dict(d):
    """Move all tensors in detection dict to CPU and detach."""
    return {
        k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in d.items()
    }


def postprocess(
    pred_tensor, gt_bboxes, gt_cls, batch_idx, img_h, img_w, conf=0.25, iou=0.45
):
    bs = pred_tensor.shape[0]
    preds_list = []
    targets_list = []

    for i in range(bs):
        p = pred_tensor[i]
        boxes_xywh = p[:4]
        cls_scores = p[4:]

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

        gt_mask = batch_idx == i
        gb = gt_bboxes[gt_mask]
        gc = gt_cls[gt_mask]
        gx1 = (gb[:, 0] - gb[:, 2] / 2) * img_w
        gy1 = (gb[:, 1] - gb[:, 3] / 2) * img_h
        gx2 = (gb[:, 0] + gb[:, 2] / 2) * img_w
        gy2 = (gb[:, 1] + gb[:, 3] / 2) * img_h
        gt_boxes_xyxy = torch.stack([gx1, gy1, gx2, gy2], dim=1)
        targets_list.append({"boxes": gt_boxes_xyxy, "labels": gc})

    return preds_list, targets_list


def evaluate_segment(
    trt_infer, val_loader, device, img_h, img_w, conf, iou, start_batch, end_batch
):
    """Evaluate a segment of batches and return preds/targets lists (on CPU)."""
    if not _HAS_TORCHMETRICS:
        return [], []

    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    local_preds = []
    local_targets = []

    with torch.no_grad():
        for batch_i, batch_data in enumerate(val_loader):
            if batch_i < start_batch:
                continue
            if batch_i >= end_batch:
                break

            imgs = batch_data["img"].to(device).float() / 255.0
            batch_idx = batch_data["batch_idx"].to(device)
            gt_bboxes = batch_data["bboxes"].to(device)
            gt_cls = batch_data["cls"].to(device).long().squeeze(-1)

            input_np = imgs.cpu().numpy().astype(np.float32)
            out_np = trt_infer.infer(input_np)
            pred_tensor = torch.from_numpy(out_np).to(device)

            preds_list, targets_list = postprocess(
                pred_tensor, gt_bboxes, gt_cls, batch_idx, img_h, img_w, conf, iou
            )

            # Move to CPU immediately to free GPU memory
            for p, t in zip(preds_list, targets_list):
                local_preds.append(to_cpu_dict(p))
                local_targets.append(to_cpu_dict(t))

            # Explicit cleanup
            del imgs, batch_idx, gt_bboxes, gt_cls, input_np, out_np, pred_tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return local_preds, local_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="Path to .engine file")
    parser.add_argument(
        "--data", default=str(_HERE / "datasets" / "dair_v2x_yolo" / "dair_v2x.yaml")
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--workers", type=int, default=0, help="Jetson: keep 0 to avoid OOM"
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--segment-size",
        type=int,
        default=500,
        help="Images per segment before mAP compute+reset",
    )
    args = parser.parse_args()

    if not _HAS_TRT:
        sys.exit(1)

    print("=" * 60)
    print(f"Engine       : {args.engine}")
    print(f"Data         : {args.data}")
    print(f"Batch        : {args.batch}")
    print(f"Segment size : {args.segment_size} imgs")
    print("=" * 60)

    trt_infer = TRTInference(args.engine)
    val_loader = get_val_loader(args.data, args.imgsz, args.batch, args.workers)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    img_h, img_w = args.imgsz, args.imgsz
    total_batches = len(val_loader)
    segment_batches = max(1, args.segment_size // args.batch)

    if not _HAS_TORCHMETRICS:
        print("[WARN] torchmetrics unavailable. Running dummy inference only.")
        with torch.no_grad():
            for batch_i, batch_data in enumerate(val_loader):
                imgs = batch_data["img"].to(device).float() / 255.0
                input_np = imgs.cpu().numpy().astype(np.float32)
                _ = trt_infer.infer(input_np)
                if (batch_i + 1) % 100 == 0:
                    print(f"  Processed {batch_i + 1}/{total_batches} batches")
        return

    # Segment-based evaluation to avoid OOM
    all_map50 = []
    all_map5095 = []

    for seg_start in range(0, total_batches, segment_batches):
        seg_end = min(seg_start + segment_batches, total_batches)
        print(f"\n[Segment] Batches {seg_start} ~ {seg_end} / {total_batches}")

        preds_seg, targets_seg = evaluate_segment(
            trt_infer,
            val_loader,
            device,
            img_h,
            img_w,
            args.conf,
            args.iou,
            seg_start,
            seg_end,
        )

        metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
        metric.update(preds_seg, targets_seg)
        results = metric.compute()
        map50 = results["map_50"].item()
        map5095 = results["map"].item()
        all_map50.append(map50)
        all_map5095.append(map5095)

        print(f"  Segment mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}")

        # Aggressive cleanup
        del preds_seg, targets_seg, metric, results
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final aggregated stats
    print("\n" + "=" * 60)
    print("TensorRT Engine mAP Results (Segment Averaged)")
    print("=" * 60)
    print(f"Segments     : {len(all_map50)}")
    print(f"mAP50  (avg) : {sum(all_map50) / len(all_map50):.4f}")
    print(f"mAP50-95(avg): {sum(all_map5095) / len(all_map5095):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
