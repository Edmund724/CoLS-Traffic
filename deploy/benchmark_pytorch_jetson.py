#!/usr/bin/env python3
"""
PyTorch FP32 推理测速脚本（Jetson 用）。
用于作为"未优化通用部署方案"的 FPS 基线，与 TensorRT INT8 对比。

Usage:
    python benchmark_pytorch_jetson.py \
        --weights best_ema.pt \
        --batch 1 --iterations 200 --warmup 50

Output:
    mean latency (ms), std (ms), FPS, throughput (img/s)
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

_HERE = Path(__file__).resolve().parent
_DETECTION_DIR = _HERE / "detection"
sys.path.insert(0, str(_DETECTION_DIR))

from nas_yolo import NASYOLOv8


def benchmark(model, device, batch_size: int, img_size: int, iterations: int, warmup: int):
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()

    # Warm-up
    print(f"[Warm-up] {warmup} iterations...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    print(f"[Benchmark] {iterations} iterations...")
    timings = []
    with torch.no_grad():
        for i in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    timings = np.array(timings)
    mean_ms = timings.mean()
    std_ms = timings.std()
    fps = batch_size / (mean_ms / 1000.0)

    print("\n" + "=" * 50)
    print("PyTorch FP32 Benchmark Results")
    print("=" * 50)
    print(f"Device        : {device}")
    print(f"Batch Size    : {batch_size}")
    print(f"Input Size    : {img_size}x{img_size}")
    print(f"Iterations    : {iterations}")
    print(f"Mean Latency  : {mean_ms:.3f} ms")
    print(f"Std Dev       : {std_ms:.3f} ms")
    print(f"FPS           : {fps:.2f}")
    print(f"Throughput    : {fps:.2f} img/s")
    print("=" * 50)
    return mean_ms, std_ms, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Load] Loading model from {args.weights}")
    model = NASYOLOv8.load(args.weights).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Info] Model params: {n_params:.2f} M")

    benchmark(model, device, args.batch, args.imgsz, args.iterations, args.warmup)


if __name__ == "__main__":
    main()
