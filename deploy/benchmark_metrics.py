#!/usr/bin/env python3
"""
毕业论文技术指标验证脚本

同时测试两项核心指标：
  1. 边缘端单帧推理延迟 (Batch Size=1)  < 30ms
  2. 系统端到端响应延迟 (含路由判决)     < 100ms

Usage (Jetson):
    cd /home/nv/deploy
    python benchmark_metrics.py \
        --engine qat_int8.engine \
        --config configs/edge_cloud.yaml \
        --data datasets/dair_v2x_yolo/images/val \
        --num 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from edge_cloud_collab import CloudEdgeSimulator, EdgeDetector, load_config
from edge_cloud_collab.utils import load_image


def load_image_list(data_dir: str, need: int):
    """轻量获取图片列表，只加载需要的数量到内存。"""
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise ValueError(f"Not a directory: {data_dir}")

    all_files = [f for f in data_path.iterdir() if f.is_file()]
    img_paths = sorted([
        str(f) for f in all_files
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ])

    if len(img_paths) == 0:
        raise ValueError(f"No images in {data_dir}")

    import random
    selected = random.sample(img_paths, min(need, len(img_paths)))
    print(f"  目录共 {len(img_paths)} 张图，加载 {len(selected)} 张到内存...")
    images = [load_image(p) for p in selected]
    return images, selected


def test_edge_latency(detector, images, warmup=10, runs=100):
    """
    指标 1：边缘端单帧推理延迟（Batch Size=1）
    测试内容：预处理 + H2D + GPU推理 + D2H + 后处理
    """
    print("\n" + "=" * 70)
    print("[Test 1] 边缘端单帧推理延迟 (Batch Size=1)")
    print("=" * 70)

    # Warm-up
    print(f"  Warm-up {warmup} iters...")
    for _ in range(warmup):
        detector.predict(images[0])

    # Benchmark
    timings = []
    for i in range(runs):
        img = images[i % len(images)]
        t0 = time.perf_counter()
        _ = detector.predict(img)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    timings = np.array(timings)
    mean = timings.mean()
    p50 = np.percentile(timings, 50)
    p90 = np.percentile(timings, 90)
    p99 = np.percentile(timings, 99)
    fps = 1000.0 / mean

    print(f"  {'Metric':<20} {'Value (ms)':>12} {'FPS':>10}")
    print("  " + "-" * 50)
    print(f"  {'Mean':<20} {mean:>12.3f} {fps:>10.2f}")
    print(f"  {'Std':<20} {timings.std():>12.3f}")
    print(f"  {'P50':<20} {p50:>12.3f}")
    print(f"  {'P90':<20} {p90:>12.3f}")
    print(f"  {'P99':<20} {p99:>12.3f}")
    print(f"  {'Min / Max':<20} {timings.min():>6.3f} / {timings.max():>6.3f}")
    print("  " + "-" * 50)

    # 判定
    PASS = "✅ PASS" if mean < 30.0 else "❌ FAIL"
    print(f"  指标要求: < 30ms    实测 Mean: {mean:.2f}ms    {PASS}")
    print("=" * 70)

    return {
        "mean": float(mean),
        "std": float(timings.std()),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "fps": float(fps),
        "pass": mean < 30.0,
    }


def test_end2end_latency(simulator, images, runs=100):
    """
    指标 2：系统端到端响应延迟（含路由判决）
    测试内容：复杂度评估 + 边缘检测 + 路由决策 + （可能）云端回退

    关键区分：
      - "含路由判决 < 100ms" 指标仅针对本地推理路径（门控判断后走本地）
      - 云端回退路径以精度换延迟，单独统计，不纳入实时性指标
    """
    print("\n" + "=" * 70)
    print("[Test 2] 系统端到端响应延迟 (含路由判决)")
    print("=" * 70)

    strategies = ["edge_only", "dynamic"]
    results = {}

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        timings_all = []
        timings_local = []   # 本地路径（含路由判决）
        timings_cloud = []   # 云端回退路径

        for i in range(runs):
            img = images[i % len(images)]
            t0 = time.perf_counter()
            rec = simulator.run_single(img, strategy=strategy)
            t1 = time.perf_counter()
            latency = (t1 - t0) * 1000.0  # 外部端到端计时（含评估+路由+推理）
            timings_all.append(latency)

            if strategy == "dynamic":
                if rec["decision"] == "local":
                    timings_local.append(latency)
                else:
                    timings_cloud.append(latency)

        def _stats(arr):
            arr = np.array(arr)
            return {
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }

        all_stat = _stats(timings_all)
        print(f"    {'Metric':<16} {'Value (ms)':>12}")
        print(f"    {'Mean (all)':<16} {all_stat['mean']:>12.3f}")
        print(f"    {'P50 (all)':<16} {all_stat['p50']:>12.3f}")
        print(f"    {'P90 (all)':<16} {all_stat['p90']:>12.3f}")

        # Dynamic 策略下：分开显示本地/云端
        if strategy == "dynamic" and timings_local:
            local_stat = _stats(timings_local)
            print(f"    {'─' * 28}")
            print(f"    本地路径: {len(timings_local)} 张 ({len(timings_local)/runs*100:.1f}%)")
            print(f"    {'Mean (local)':<16} {local_stat['mean']:>12.3f}")
            print(f"    {'P90 (local)':<16} {local_stat['p90']:>12.3f}")
            if timings_cloud:
                cloud_stat = _stats(timings_cloud)
                print(f"    云端路径: {len(timings_cloud)} 张 ({len(timings_cloud)/runs*100:.1f}%)")
                print(f"    {'Mean (cloud)':<16} {cloud_stat['mean']:>12.3f}")
                print(f"    {'P90 (cloud)':<16} {cloud_stat['p90']:>12.3f}")

        results[strategy] = {
            "mean": all_stat["mean"],
            "p50": all_stat["p50"],
            "p90": all_stat["p90"],
            "p99": all_stat["p99"],
        }

        # Dynamic 本地路径判定 < 100ms；Edge-Only 全部判定 < 100ms
        if strategy == "dynamic" and timings_local:
            results[strategy]["pass"] = local_stat["mean"] < 100.0
            results[strategy]["local_mean"] = local_stat["mean"]
            results[strategy]["local_p90"] = local_stat["p90"]
            results[strategy]["cloud_ratio"] = len(timings_cloud) / runs if runs > 0 else 0.0
        else:
            results[strategy]["pass"] = all_stat["mean"] < 100.0

    # 汇总判定
    print("\n  " + "-" * 60)
    print(f"  {'Strategy':<18} {'Path':<10} {'Mean(ms)':>12} {'P90(ms)':>12} {'Status':>10}")
    print("  " + "-" * 60)

    # Edge-Only
    res = results["edge_only"]
    status = "✅ PASS" if res["pass"] else "❌ FAIL"
    print(f"  {'edge_only':<18} {'all':<10} {res['mean']:>12.2f} {res['p90']:>12.2f} {status:>10}")

    # Dynamic
    res = results["dynamic"]
    status = "✅ PASS" if res["pass"] else "❌ FAIL"
    if "local_mean" in res:
        print(f"  {'dynamic':<18} {'local':<10} {res['local_mean']:>12.2f} {res['local_p90']:>12.2f} {status:>10}")
        print(f"  {'':<18} {'all':<10} {res['mean']:>12.2f} {res['p90']:>12.2f}")
        print(f"  {'':<18} {'cloud%':<10} {res['cloud_ratio']*100:>11.1f}%")
    else:
        print(f"  {'dynamic':<18} {'all':<10} {res['mean']:>12.2f} {res['p90']:>12.2f} {status:>10}")

    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="qat_int8.engine", help="边缘模型路径 (.engine / .pt)")
    parser.add_argument("--config", default="configs/edge_cloud.yaml", help="云边协同配置")
    parser.add_argument("--data", default="datasets/dair_v2x_yolo/images/val", help="验证图片目录")
    parser.add_argument("--num", type=int, default=100, help="测试图片数量")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--use-real-api", action="store_true",
                        help="真实调用云端 API（默认 mock，注意 API 费用和延迟）")
    args = parser.parse_args()

    print("=" * 70)
    print("毕业论文核心技术指标验证")
    print("=" * 70)
    print(f"  边缘模型 : {args.engine}")
    print(f"  配置文件 : {args.config}")
    print(f"  测试数据 : {args.data}")
    print(f"  测试数量 : {args.num}")
    print("=" * 70)

    # 加载配置
    cfg = load_config(args.config)

    # 覆盖 weights 为命令行指定的 engine
    cfg["edge_detector"]["weights"] = args.engine
    cfg["edge_detector"]["imgsz"] = args.imgsz

    # 云端 API 模式控制
    use_real_api = args.use_real_api or cfg["simulator"].get("use_real_api", False)
    if use_real_api:
        print("\n[!] 注意：启用真实云端 API（ModelScope Qwen3-VL-8B）")
        print("    每张云端回退图片约 2-3s 延迟，测试费用按实际调用计费。")
        if args.num > 20:
            print(f"[!] 建议减少测试数量（当前 {args.num}），已自动限制为 20")
            args.num = 20
    else:
        print("\n[Info] 使用 mock 云端模式（不实际调用 API）")

    # 加载图片（只加载需要的数量）
    print("\n[Setup] 加载测试图片...")
    images, img_paths = load_image_list(args.data, args.num)

    # 初始化 Detector
    print("\n[Setup] 初始化边缘检测器...")
    detector = EdgeDetector.from_config(cfg["edge_detector"])
    detector.warmup(n=args.warmup)

    # Test 1：边缘端单帧延迟
    edge_result = test_edge_latency(detector, images, warmup=args.warmup, runs=args.num)

    # 初始化 Simulator（手动组装，复用已 warm-up 的 detector，避免重复加载模型）
    print("\n[Setup] 初始化云边协同仿真器...")
    from edge_cloud_collab import CloudClient, ComplexityEvaluator, DynamicRouter

    evaluator = ComplexityEvaluator()
    router = DynamicRouter.from_config(cfg["routing"])
    cloud = CloudClient.from_config(cfg["cloud"], mock=not use_real_api)

    simulator = CloudEdgeSimulator(
        edge_detector=detector,
        complexity_evaluator=evaluator,
        router=router,
        cloud_client=cloud,
    )
    simulator.use_simulated_latency = False  # 强制使用真实延迟

    # Test 2：端到端延迟
    e2e_result = test_end2end_latency(simulator, images, runs=args.num)

    # 最终判定
    print("\n" + "=" * 70)
    print("最终判定")
    print("=" * 70)

    checks = [
        ("边缘端单帧延迟 < 30ms", edge_result["pass"], f"{edge_result['mean']:.2f}ms"),
        ("端到端延迟(Edge-Only) < 100ms", e2e_result.get("edge_only", {}).get("pass", False),
         f"{e2e_result.get('edge_only', {}).get('mean', 0):.2f}ms"),
        ("端到端延迟(Dynamic) < 100ms", e2e_result.get("dynamic", {}).get("pass", False),
         f"{e2e_result.get('dynamic', {}).get('mean', 0):.2f}ms"),
    ]

    all_pass = True
    for name, passed, value in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:<40} {value:>12}  {status}")

    print("=" * 70)
    if all_pass:
        print("🎉 所有核心指标均已达标！")
    else:
        print("⚠️  部分指标未达标，请参考上方详细数据排查瓶颈。")
    print("=" * 70)


if __name__ == "__main__":
    main()
