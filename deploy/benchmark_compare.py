#!/usr/bin/env python3
"""
统一对比脚本：PyTorch FP32 vs TensorRT INT8 QAT 推理 FPS

只对比纯模型推理时间（GPU 执行时间），对齐测试条件：
  - 同一模型架构（NAS-YOLOv8, 2.75M params）
  - 同一输入尺寸（1, 3, 640, 640）
  - 同一 batch size（1）
  - 同一迭代次数（warmup + benchmark）

Usage:
    cd /home/nv/deploy
    python benchmark_compare.py \
        --weights best_ema.pt \
        --engine qat_int8.engine \
        --iterations 200 \
        --warmup 50

Output:
    - PyTorch FP32 GPU 时间 & FPS
    - TensorRT INT8 QAT GPU 时间 & FPS
    - 加速比 & 判定（是否 > 5x）
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


_HERE = Path(__file__).resolve().parent


def run_pytorch_benchmark(weights: str, iterations: int, warmup: int):
    """运行 PyTorch FP32 benchmark，返回 (gpu_ms, fps, raw_stdout)。"""
    cmd = [
        sys.executable, "benchmark_pytorch_jetson.py",
        "--weights", weights,
        "--batch", "1",
        "--imgsz", "640",
        "--iterations", str(iterations),
        "--warmup", str(warmup),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_HERE))
    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"[ERROR] PyTorch benchmark failed:\n{stderr}", file=sys.stderr)
        return None, None, stdout + stderr

    # 解析: Mean Latency  : 45.234 ms
    mean_match = re.search(r'Mean Latency\s*:\s*([\d.]+)\s*ms', stdout)
    # 解析: FPS           : 22.11
    fps_match = re.search(r'FPS\s*:\s*([\d.]+)', stdout)

    mean_ms = float(mean_match.group(1)) if mean_match else None
    fps = float(fps_match.group(1)) if fps_match else None
    return mean_ms, fps, stdout


def run_trt_benchmark(engine: str, iterations: int, warmup: int):
    """运行 TensorRT benchmark，返回 (gpu_ms, fps, raw_stdout)。"""
    # 找一张测试图（test.jpg 或 val 目录下任意一张）
    test_img = _HERE / "test.jpg"
    if not test_img.exists():
        val_dir = _HERE / "datasets" / "dair_v2x_yolo" / "images" / "val"
        if val_dir.exists():
            imgs = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
            if imgs:
                test_img = imgs[0]
    if not test_img.exists():
        print("[ERROR] No test image found. Please provide test.jpg or ensure val images exist.",
              file=sys.stderr)
        return None, None, ""

    cmd = [
        sys.executable, "trt_runtime.py",
        "--engine", engine,
        "--input", str(test_img),
        "--benchmark", str(iterations + warmup),  # trt_runtime 内部自行 warmup
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_HERE))
    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print(f"[ERROR] TRT benchmark failed:\n{stderr}", file=sys.stderr)
        return None, None, stdout + stderr

    # 解析 breakdown 表格中 GPU 行的 Mean 值
    # 格式示例:
    #   Stage            Mean(ms)    Std(ms)     P50        P90        P99       Ratio
    #   ------------------------------------------------------------------------------
    #   preprocess         7.61       0.52       7.58       8.12       8.89      25.6%
    #   GPU               11.20       0.45      11.15      11.78      12.34      37.6%
    gpu_match = re.search(r'^\s*GPU\s+([\d.]+)', stdout, re.MULTILINE)

    gpu_ms = float(gpu_match.group(1)) if gpu_match else None
    fps = 1000.0 / gpu_ms if gpu_ms else None
    return gpu_ms, fps, stdout


def print_results(pt_mean, pt_fps, trt_mean, trt_fps, save_path=None):
    """打印对比表格。"""
    speedup = pt_mean / trt_mean if (pt_mean and trt_mean) else 0.0
    fps_ratio = trt_fps / pt_fps if (pt_fps and trt_fps) else 0.0

    print("\n" + "=" * 70)
    print("推理 FPS 对比结果（纯 GPU 执行时间）")
    print("=" * 70)
    print(f"{'框架':<22} {'GPU 时间(ms)':>12} {'FPS':>10} {'vs 基线':>12}")
    print("-" * 70)
    print(f"{'PyTorch FP32':<22} {pt_mean:>12.2f} {pt_fps:>10.2f} {'1.00x':>12}")
    print(f"{'TensorRT INT8 QAT':<22} {trt_mean:>12.2f} {trt_fps:>10.2f} {speedup:>11.2f}x")
    print("-" * 70)

    status = "✅ PASS (>5x)" if speedup >= 5 else f"❌ FAIL ({speedup:.2f}x < 5x)"
    print(f"\n  加速比 (GPU time) : {speedup:.2f}x")
    print(f"  FPS 提升倍数      : {fps_ratio:.2f}x")
    print(f"  指标判定 (>5x)    : {status}")
    print("=" * 70)

    # 保存 JSON
    if save_path:
        data = {
            "pytorch_fp32": {
                "gpu_ms": round(pt_mean, 3),
                "fps": round(pt_fps, 2),
            },
            "trt_int8_qat": {
                "gpu_ms": round(trt_mean, 3),
                "fps": round(trt_fps, 2),
            },
            "speedup": round(speedup, 2),
            "fps_ratio": round(fps_ratio, 2),
            "pass_5x": speedup >= 5,
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[Saved] 结果已保存到 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch vs TensorRT FPS 对比")
    parser.add_argument("--weights", default="best_ema.pt",
                        help="PyTorch 权重路径 (.pt)")
    parser.add_argument("--engine", default="qat_int8.engine",
                        help="TensorRT engine 路径")
    parser.add_argument("--iterations", type=int, default=200,
                        help="benchmark 迭代次数")
    parser.add_argument("--warmup", type=int, default=50,
                        help="warmup 迭代次数")
    parser.add_argument("--save", default="results/fps_compare.json",
                        help="结果保存路径 (JSON)")
    args = parser.parse_args()

    print("=" * 70)
    print("PyTorch FP32 vs TensorRT INT8 QAT — 推理 FPS 统一对比")
    print("=" * 70)
    print(f"PyTorch 权重 : {args.weights}")
    print(f"TRT Engine   : {args.engine}")
    print(f"迭代次数     : {args.iterations} (warmup {args.warmup})")
    print(f"输入尺寸     : 1 x 3 x 640 x 640")
    print(f"对比范围     : 纯 GPU 执行时间（不含预处理/后处理）")
    print("=" * 70)

    # PyTorch
    print("\n[1/2] 运行 PyTorch FP32 benchmark...")
    pt_mean, pt_fps, pt_out = run_pytorch_benchmark(
        args.weights, args.iterations, args.warmup
    )
    if pt_mean is None:
        print("PyTorch benchmark 失败，退出。")
        sys.exit(1)
    print(f"  → GPU 时间: {pt_mean:.2f} ms, FPS: {pt_fps:.2f}")

    # TensorRT
    print("\n[2/2] 运行 TensorRT INT8 QAT benchmark...")
    trt_mean, trt_fps, trt_out = run_trt_benchmark(
        args.engine, args.iterations, args.warmup
    )
    if trt_mean is None:
        print("TRT benchmark 失败，退出。")
        sys.exit(1)
    print(f"  → GPU 时间: {trt_mean:.2f} ms, FPS: {trt_fps:.2f}")

    # 输出对比
    print_results(pt_mean, pt_fps, trt_mean, trt_fps, save_path=args.save)


if __name__ == "__main__":
    main()
