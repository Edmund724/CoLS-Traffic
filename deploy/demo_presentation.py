#!/usr/bin/env python3
"""
系统演示 Demo 编排脚本

在 Jetson 上按顺序真实运行所有推理环节（含云端 API 调用），
配合屏幕录制产出答辩 Demo 视频。后期剪辑掉等待间隙即可。

Usage (Jetson):
    cd /home/nv/deploy
    python demo_presentation.py

    # 跳过 benchmark，直接展示总结表（基于已有 JSON 结果调试）
    python demo_presentation.py --skip-benchmark

    # 减少路由演示图片数（节省 API 调用）
    python demo_presentation.py --routing-num 10

录制提示：
    - 终端字体调至 18pt+，全屏显示
    - 屏幕录制后剪掉 engine 加载、API 等待等间隙
    - 原始录制可能 10-15 分钟，剪辑目标 5 分钟
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ── 终端输出工具 ──────────────────────────────────────────────────────────

WIDTH = 78


def print_banner(title: str):
    """打印大标题分隔线（方便后期剪辑定位）。"""
    print("\n" + "=" * WIDTH)
    print(f"  ▶ {title}")
    print("=" * WIDTH)


def print_section(title: str):
    """打印小节分隔线。"""
    print("\n" + "─" * WIDTH)
    print(f"  {title}")
    print("─" * WIDTH)


# ── 模块1：推理加速对比 ───────────────────────────────────────────────────

MODULE1_SAVE = _HERE / "results" / "fps_compare.json"


def module1_speedup(skip: bool = False):
    """PyTorch FP32 vs TensorRT INT8 QAT 推理加速对比。"""
    print_banner("模块一：Jetson 推理加速对比 (PyTorch FP32 vs TensorRT INT8 QAT)")

    if skip:
        print("[Skip] 跳过 benchmark，加载已有结果...")
        return _load_json(MODULE1_SAVE)

    # 自动预热：初始化 EdgeDetector，让 TensorRT engine 进入热状态
    print("  [Warmup] 正在预热 TensorRT engine...")
    sys.path.insert(0, str(_HERE))
    from edge_cloud_collab import EdgeDetector

    detector = EdgeDetector(weights="qat_int8.engine")
    detector.warmup(n=3)
    print("  [Warmup] Engine 预热完成")

    cmd = [
        sys.executable,
        "benchmark_compare.py",
        "--weights",
        "best_ema.pt",
        "--engine",
        "qat_int8.engine",
        "--iterations",
        "200",
        "--warmup",
        "50",
        "--save",
        str(MODULE1_SAVE),
    ]

    print(f"\n  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_HERE))

    # 选择性打印关键输出（过滤掉冗长的中间过程）
    _print_selective(
        result.stdout,
        [
            "对比结果",
            "框架",
            "PyTorch",
            "TensorRT",
            "加速比",
            "FPS",
            "PASS",
            "FAIL",
            "=" * 10,
        ],
    )

    if result.returncode != 0:
        print(f"\n[ERROR] benchmark_compare.py 失败:\n{result.stderr}")
        return None

    return _load_json(MODULE1_SAVE)


# ── 模块2：动态路由演示 ───────────────────────────────────────────────────


def module2_routing(skip: bool = False, num_images: int = 20):
    """云边协同动态路由批量演示。"""
    print_banner("模块二：云边协同动态路由演示")

    if skip:
        print("[Skip] 跳过动态路由演示")
        return None

    sys.path.insert(0, str(_HERE))
    from edge_cloud_collab import CloudEdgeSimulator, load_config

    config_path = _HERE / "configs" / "edge_cloud.yaml"
    cfg = load_config(str(config_path))

    data_path = Path(cfg["dataset"]["data_yaml"]).parent
    val_dir = data_path / "images" / "val"
    train_dir = data_path / "images" / "train"
    image_dir = val_dir if val_dir.exists() else train_dir

    all_images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if not all_images:
        print(f"[ERROR] 未找到图像: {image_dir}")
        return None

    # 固定随机种子，保证演示可复现
    random.seed(42)
    selected = random.sample(all_images, min(num_images, len(all_images)))

    print(f"\n  数据集: {cfg['dataset']['data_yaml']}")
    print(f"  图像总数: {len(all_images)}, 本次演示: {len(selected)} 张")

    sim = CloudEdgeSimulator.from_config(str(config_path))

    print_section(
        f"运行动态路由策略 — {len(selected)} 张图"
    )
    print("  （回退到云端的样本会真实调用 VLM API）\n")

    results = sim.compare_strategies([str(p) for p in selected])

    # 提取 Dynamic 策略关键指标供总结表使用
    dyn = results.get("dynamic", {})
    summary = {}
    if "latency_ms" in dyn:
        local_e2e = dyn.get("local_e2e_ms", {})
        summary = {
            "local_e2e_mean": local_e2e.get("mean", 999.0),
            "local_e2e_p90": local_e2e.get("p90", 999.0),
            "num_local": dyn["num_local"],
            "num_cloud": dyn["num_cloud"],
            "total": dyn["total_images"],
            "cloud_ratio": dyn["num_cloud"] / dyn["total_images"]
            if dyn["total_images"]
            else 0,
        }

    # 将 summary 附加到 results 中方便后续读取
    results["_summary"] = summary
    return results


# ── 模块3：协同纠错（真实 API） ────────────────────────────────────────────

COMPLEX_SCENES = _HERE / "results" / "complex_scenes_500.json"


def _get_complex_scenes(args_path: str | None) -> Path:
    """返回复杂场景子集路径。"""
    if args_path:
        return Path(args_path)
    return COMPLEX_SCENES


def module3_coop(skip: bool = False, num: int = 100, complex_scenes_path: str = ""):
    """大模型协同纠错准确率提升验证（真实云端 API）。"""
    print_banner("模块三：大模型协同纠错准确率提升验证（真实云端 API）")

    complex_scenes_path = _get_complex_scenes(complex_scenes_path)
    if not complex_scenes_path.exists():
        print(f"\n  [Auto] 缓存不存在，自动筛选复杂场景子集...")
        complex_scenes_path.parent.mkdir(parents=True, exist_ok=True)
        filter_cmd = [
            sys.executable,
            "benchmark_accuracy_coop.py",
            "--engine",
            "qat_int8.engine",
            "--data",
            "datasets/dair_v2x_yolo",
            "--split",
            "val",
            "--complexity-ratio",
            "0.3",
            "--num",
            "100",
            "--save-complex-scenes",
            str(complex_scenes_path),
        ]
        filter_result = subprocess.run(
            filter_cmd, capture_output=True, text=True, cwd=str(_HERE)
        )
        if filter_result.returncode != 0 or not complex_scenes_path.exists():
            print(f"\n[ERROR] 自动筛选失败:\n{filter_result.stderr}")
            return None
        print(f"  [Auto] 复杂场景子集已保存: {complex_scenes_path}")

    if skip:
        print("[Skip] 跳过协同纠错 benchmark")
        return None

    cmd = [
        sys.executable,
        "benchmark_accuracy_coop.py",
        "--engine",
        "qat_int8.engine",
        "--data",
        "datasets/dair_v2x_yolo",
        "--load-complex-scenes",
        str(complex_scenes_path),
        "--num",
        str(num),
    ]

    print(f"\n  Command: {' '.join(cmd)}")

    # 流式执行：实时打印 stdout，同时收集完整输出用于后续解析
    # PYTHONUNBUFFERED=1 强制子进程 stdout 行缓冲，避免输出一段一段积累
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(_HERE),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    stdout_lines = []
    for line in process.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        stdout_lines.append(line)

    process.wait()
    stdout = "\n".join(stdout_lines)

    if process.returncode != 0:
        print(
            f"\n[ERROR] benchmark_accuracy_coop.py 失败（exit code={process.returncode}）"
        )
        return None

    # 解析关键指标（仅 Precision）
    parsed = {
        "edge_precision": None,
        "coop_precision": None,
        "boost_precision": None,
    }
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if "Precision@0.5:" in line:
            # 下一行是边缘-only
            if i + 1 < len(lines) and "边缘-only" in lines[i + 1]:
                parsed["edge_precision"] = _extract_float(lines[i + 1])
            if i + 2 < len(lines) and "协同纠错" in lines[i + 2]:
                parsed["coop_precision"] = _extract_float(lines[i + 2])
            if i + 3 < len(lines) and "相对提升" in lines[i + 3]:
                m = re.search(r"([+-]?\d+\.?\d*)%", lines[i + 3])
                if m:
                    parsed["boost_precision"] = float(m.group(1))

    # 打印关键指标
    print_section("协同纠错关键指标")
    if parsed["edge_precision"] is not None:
        print(f"  Precision@0.5  边缘-only : {parsed['edge_precision']:.4f}")
        print(f"  Precision@0.5  协同纠错  : {parsed['coop_precision']:.4f}")
        print(f"  Precision@0.5  相对提升  : {parsed['boost_precision']:+.1f}%")

    return parsed


# ── 模块4：总结表 ─────────────────────────────────────────────────────────


def module4_summary(res_speedup, res_routing, res_coop):
    """打印技术指标达成总览表。"""
    print_banner("模块四：技术指标达成总览（真实运行）")

    # --- 指标1：FPS 加速比 ---
    speedup = 0.0
    if res_speedup and "speedup" in res_speedup:
        speedup = res_speedup["speedup"]
    speedup_ok = speedup >= 5.0

    # --- 指标2：单帧 GPU 延迟 ---
    gpu_ms = 999.0
    if res_speedup and "trt_int8_qat" in res_speedup:
        gpu_ms = res_speedup["trt_int8_qat"].get("gpu_ms", 999.0)
    latency_ok = gpu_ms < 30.0

    # --- 指标3：端到端延迟（本地推理样本，含路由判决）---
    e2e_ms = 999.0
    if res_routing and "_summary" in res_routing:
        e2e_ms = res_routing["_summary"].get("local_e2e_mean", 999.0)
    e2e_ok = e2e_ms < 100.0

    # --- 指标4：协同准确率提升 ---
    boost = -999.0
    if res_coop and res_coop.get("boost_precision") is not None:
        boost = res_coop["boost_precision"]
    coop_ok = boost > 10.0

    # --- 打印表格 ---
    print()
    print(f"  {'指标':<18} {'要求':<14} {'实测值':<20} {'状态':<10}")
    print(f"  {'─' * 62}")
    _print_row("FPS 加速比", "> 5x", f"{speedup:.2f}x", speedup_ok)
    _print_row("单帧 GPU 延迟", "< 30ms", f"{gpu_ms:.2f} ms", latency_ok)
    _print_row("端到端延迟", "< 100ms", f"{e2e_ms:.1f} ms", e2e_ok)
    _print_row("协同准确率提升", "> 10%", f"{boost:+.1f}%", coop_ok)
    print(f"  {'─' * 62}")

    all_ok = all([speedup_ok, latency_ok, e2e_ok, coop_ok])
    if all_ok:
        print(f"\n  🎉 所有技术指标全部达成！")
    else:
        print(f"\n  ⚠️  部分指标未达成，请检查上述 FAIL 项。")
    print("=" * WIDTH)

    # 保存 JSON 供后续使用
    summary_data = {
        "speedup": {"value": speedup, "target": "> 5x", "pass": speedup_ok},
        "gpu_latency_ms": {"value": gpu_ms, "target": "< 30ms", "pass": latency_ok},
        "local_e2e_latency_ms": {"value": e2e_ms, "target": "< 100ms", "pass": e2e_ok},
        "coop_boost_precision_percent": {
            "value": boost,
            "target": "> 10%",
            "pass": coop_ok,
        },
        "all_pass": all_ok,
    }
    save_path = _HERE / "results" / "demo_summary.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n  [Saved] 总结已保存: {save_path}")


def _print_row(name: str, target: str, value: str, ok: bool):
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {name:<18} {target:<14} {value:<20} {status:<10}")


# ── 工具函数 ──────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _print_selective(stdout: str, keywords: list[str]):
    """只打印包含关键词的行，过滤掉冗长中间输出。"""
    for line in stdout.splitlines():
        if any(kw in line for kw in keywords):
            print(line)


def _extract_float(line: str) -> float | None:
    """从一行文本中提取第一个浮点数。"""
    m = re.search(r"([\d.]+)", line)
    return float(m.group(1)) if m else None


# ── 主入口 ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="系统演示 Demo 编排脚本 — Jetson 真实运行版"
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="跳过所有 benchmark，直接展示总结表（基于已有 JSON 结果调试）",
    )
    parser.add_argument(
        "--routing-num",
        type=int,
        default=100,
        help="动态路由演示的图像数量 (default: 100)",
    )
    parser.add_argument(
        "--coop-num",
        type=int,
        default=100,
        help="协同纠错验证的图像数量 (default: 100)",
    )
    parser.add_argument(
        "--complex-scenes",
        type=str,
        default="",
        help="预筛选的复杂场景子集 JSON 路径 (default: results/complex_scenes_500.json)",
    )
    args = parser.parse_args()

    print("=" * WIDTH)
    print("  系统演示 Demo — 云边协同动态路由系统")
    print("=" * WIDTH)

    # 顺序执行 4 个模块
    res_speedup = module1_speedup(skip=args.skip_benchmark)
    res_routing = module2_routing(
        skip=args.skip_benchmark,
        num_images=args.routing_num,
    )
    res_coop = module3_coop(
        skip=args.skip_benchmark,
        num=args.coop_num,
        complex_scenes_path=args.complex_scenes,
    )
    module4_summary(res_speedup, res_routing, res_coop)


if __name__ == "__main__":
    main()
