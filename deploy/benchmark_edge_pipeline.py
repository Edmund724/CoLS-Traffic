#!/usr/bin/env python3
"""
边缘端 TensorRT 批量推理测速脚本（Jetson 用）。
对比串行推理 vs Double-buffer Pipeline 的 throughput。

Usage:
    cd /home/nv/deploy
    python benchmark_edge_pipeline.py \
        --engine qat_int8.engine \
        --data datasets/dair_v2x_yolo/images/val \
        --num 100

Output:
    - 单帧端到端延迟分解（pre / infer / post）
    - 串行批量 throughput
    - Pipeline 批量 throughput
"""

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

# 把 edge_cloud_collab 加入路径
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from edge_cloud_collab import EdgeDetector
from edge_cloud_collab.utils import load_image


def benchmark_single(detector, images, warmup=10, runs=100):
    """串行单帧推理测速（端到端）。"""
    # Warm-up
    print(f"[Warm-up] {warmup} iterations...")
    for _ in range(warmup):
        _ = detector.predict(images[0])

    # Benchmark
    print(f"[Benchmark-Serial] {runs} iterations...")
    timings = []
    for i in range(runs):
        img = images[i % len(images)]
        t0 = time.perf_counter()
        _ = detector.predict(img)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    timings = np.array(timings)
    print("\n" + "=" * 60)
    print("Serial Inference (End-to-End)")
    print("=" * 60)
    print(f"  Mean    : {timings.mean():.3f} ms")
    print(f"  Std     : {timings.std():.3f} ms")
    print(f"  P50     : {np.percentile(timings, 50):.3f} ms")
    print(f"  P90     : {np.percentile(timings, 90):.3f} ms")
    print(f"  P99     : {np.percentile(timings, 99):.3f} ms")
    print(f"  Min/Max : {timings.min():.3f} / {timings.max():.3f} ms")
    print(f"  FPS     : {1000.0 / timings.mean():.2f}")
    print("=" * 60)
    return timings


def benchmark_pipeline(detector, images, num_runs=100):
    """Double-buffer Pipeline 批量推理测速。"""
    if not detector.is_trt:
        print("[Skip] Pipeline benchmark only available for TensorRT backend")
        return None

    print(f"[Benchmark-Pipeline] {num_runs} images with double-buffer...")
    # 循环使用 images，凑够 num_runs 张
    extended = [images[i % len(images)] for i in range(num_runs)]
    results = detector.predict_pipeline(extended)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="qat_int8.engine", help="TensorRT engine path")
    parser.add_argument("--data", default="datasets/dair_v2x_yolo/images/val",
                        help="Image directory or single image")
    parser.add_argument("--num", type=int, default=100, help="Number of images for benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    print("=" * 60)
    print("Edge Detector Throughput Benchmark")
    print("=" * 60)
    print(f"Engine : {args.engine}")
    print(f"Data   : {args.data}")
    print(f"Images : {args.num}")
    print("=" * 60)

    # 1. 初始化 detector
    detector = EdgeDetector(
        weights=args.engine,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
    )
    print(f"Backend: {'TensorRT' if detector.is_trt else 'PyTorch'}")

    # 2. 轻量获取图片列表（只读文件名，不加载内容）
    data_path = Path(args.data)
    if data_path.is_dir():
        # 用 os.listdir 比 glob 更快，目录项多时也不会卡
        all_files = [f for f in data_path.iterdir() if f.is_file()]
        img_paths = sorted([
            str(f) for f in all_files
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ])
    else:
        img_paths = [str(data_path)]

    if len(img_paths) == 0:
        print(f"[ERROR] No images found in {args.data}")
        sys.exit(1)

    print(f"Directory has {len(img_paths)} images total.")

    # 只加载 benchmark 需要的数量，避免内存爆炸
    need = min(args.num, len(img_paths))
    selected_paths = img_paths[:need]
    print(f"Loading {need} images into memory for benchmark...")
    images = [load_image(p) for p in selected_paths]
    print(f"Loaded {len(images)} images. Resolutions: {[img.shape[:2] for img in images[:3]]}...")

    # 3. Warm-up
    detector.warmup(n=args.warmup)

    # 4. 串行测速
    serial_times = benchmark_single(detector, images, warmup=args.warmup, runs=args.num)

    # 5. Pipeline 测速（仅 TRT）
    if detector.is_trt:
        print()
        benchmark_pipeline(detector, images, num_runs=args.num)

    # 6. 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    mean_ms = serial_times.mean()
    print(f"Serial end-to-end : {mean_ms:.2f} ms ({1000.0/mean_ms:.1f} FPS)")
    if detector.is_trt:
        print(f"Pipeline throughput: see predict_pipeline output above")
    print("=" * 60)


if __name__ == "__main__":
    main()
