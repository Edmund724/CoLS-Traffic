#!/usr/bin/env python3
"""
动态路由阈值标定脚本

基于 DAIR-V2X val 集真实数据统计 13 维特征分布，
并给出 force_cloud / force_local 的阈值建议。

Usage (Jetson):
    cd /home/nv/deploy
    python calibrate_routing_thresholds.py \
        --engine qat_int8.engine \
        --data datasets/dair_v2x_yolo \
        --split val \
        --output reports/routing_threshold_calibration.json

注意：本脚本需在 Jetson Orin Nano 上运行（依赖 TensorRT + pycuda）。
当前服务器环境缺少 Jetson 编译的 .engine 所需的 pycuda，无法直接运行。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from edge_cloud_collab import EdgeDetector, ComplexityEvaluator
from edge_cloud_collab.utils import load_image


def parse_args():
    parser = argparse.ArgumentParser(description="标定动态路由阈值")
    parser.add_argument("--engine", default="qat_int8.engine", help="TensorRT engine 路径")
    parser.add_argument("--data", default="../datasets/dair_v2x_yolo", help="数据集根目录")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="数据集划分")
    parser.add_argument("--output", default="reports/routing_threshold_calibration.json",
                        help="输出 JSON 路径")
    parser.add_argument("--max-images", type=int, default=0,
                        help="最大处理图片数（0=全部）")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def get_image_label_pairs(data_dir: Path, split: str):
    """获取图片和标注的配对列表。"""
    img_dir = data_dir / "images" / split
    label_dir = data_dir / "labels" / split

    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    pairs = []
    for img_path in img_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))
    return pairs


def extract_features(img_path: Path, detector, evaluator):
    """对单张图片提取 13 维特征。"""
    img_rgb = load_image(str(img_path))
    h, w = img_rgb.shape[:2]

    # 图像级特征
    img_feats = evaluator.image_features(img_rgb)

    # 边缘检测
    edge_result = detector.predict(img_rgb)
    edge_dets = edge_result["detections"]

    # 检测级特征
    det_feats = evaluator.detection_features(edge_dets, (h, w))

    # 合并（保留原始值 _raw）
    features = {**img_feats, **det_feats}
    return features, edge_dets


def compute_stats(values: list[float]) -> dict:
    """计算统计量。"""
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def suggest_thresholds(stats: dict[str, dict]) -> dict:
    """基于统计分布生成阈值建议（修正版）。"""
    suggestions = {
        "force_cloud": {},
        "force_local": {},
        "rationale": {},
    }

    # ── force_cloud 建议 ──
    # 模糊: 拉普拉斯方差 < P10（最模糊的 10%）
    suggestions["force_cloud"]["blur_laplacian_max"] = {
        "value": round(stats["laplacian_var_raw"]["p10"], 1),
        "rationale": f"P10={stats['laplacian_var_raw']['p10']:.1f}, 最模糊的10%",
    }

    # 过曝/欠曝: 在 DAIR-V2X 中几乎不存在（P90 < 0.001），不设为 force_cloud
    # 若真实场景有夜间/逆光需求，可手动加入
    max_oe = stats["overexposure_ratio_raw"]["max"]
    max_ue = stats["underexposure_ratio_raw"]["max"]
    if max_oe > 0.01:
        suggestions["force_cloud"]["overexposure_min"] = {
            "value": round(max_oe * 0.5, 3),
            "rationale": f"数据集max={max_oe:.4f}, 取一半",
        }
    if max_ue > 0.05:
        suggestions["force_cloud"]["underexposure_min"] = {
            "value": round(max_ue * 0.5, 3),
            "rationale": f"数据集max={max_ue:.4f}, 取一半",
        }

    # 低置信度比例: > P90（最不确定的 10%，而非 P75 的 25%）
    suggestions["force_cloud"]["low_conf_ratio_min"] = {
        "value": round(stats["low_conf_ratio_raw"]["p90"], 3),
        "rationale": f"P90={stats['low_conf_ratio_raw']['p90']:.3f}, 低conf最多的10%",
    }

    # 平均置信度: < P10（最低的 10%，而非 P25 的 25%）
    suggestions["force_cloud"]["mean_confidence_max"] = {
        "value": round(stats["mean_confidence_raw"]["p10"], 3),
        "rationale": f"P10={stats['mean_confidence_raw']['p10']:.3f}, 最不确定的10%",
    }

    # 目标密度: > P90（最密集的 10%）
    suggestions["force_cloud"]["object_density_max"] = {
        "value": round(stats["object_density_raw"]["p90"], 3),
        "rationale": f"P90={stats['object_density_raw']['p90']:.3f}, 最密集的10%",
    }

    # ── force_local 建议 ──
    # 空旷: 目标密度 < P25（最空旷的 25%，而非 P10 的 10%）
    suggestions["force_local"]["object_density_max"] = {
        "value": round(stats["object_density_raw"]["p25"], 3),
        "rationale": f"P25={stats['object_density_raw']['p25']:.3f}, 最空旷的25%",
    }

    # 高度确信: 平均置信度 > P75（较确信的 25%，而非 P90 的 10%）
    suggestions["force_local"]["mean_confidence_min"] = {
        "value": round(stats["mean_confidence_raw"]["p75"], 3),
        "rationale": f"P75={stats['mean_confidence_raw']['p75']:.3f}, 较确信的25%",
    }

    # 清晰: 拉普拉斯方差 > P75（较清晰的 25%，而非 P90 的 10%）
    suggestions["force_local"]["blur_laplacian_min"] = {
        "value": round(stats["laplacian_var_raw"]["p75"], 1),
        "rationale": f"P75={stats['laplacian_var_raw']['p75']:.1f}, 较清晰的25%",
    }

    return suggestions


def simulate_routing(all_features: list[dict], thresholds: dict) -> dict:
    """用建议阈值模拟路由决策，统计回退比例。"""
    fc = thresholds["force_cloud"]
    fl = thresholds["force_local"]

    num_cloud = 0
    num_local = 0
    cloud_reasons = []

    for feats in all_features:
        decision = "local"
        reasons = []

        # force_cloud
        if feats["laplacian_var_raw"] < fc["blur_laplacian_max"]["value"]:
            decision = "cloud"; reasons.append("blur")
        elif "overexposure_min" in fc and feats["overexposure_ratio_raw"] > fc["overexposure_min"]["value"]:
            decision = "cloud"; reasons.append("overexposure")
        elif "underexposure_min" in fc and feats["underexposure_ratio_raw"] > fc["underexposure_min"]["value"]:
            decision = "cloud"; reasons.append("underexposure")
        elif feats["low_conf_ratio_raw"] > fc["low_conf_ratio_min"]["value"]:
            decision = "cloud"; reasons.append("low_conf")
        elif feats.get("object_count_norm_raw", 0) > 0 and feats["mean_confidence_raw"] < fc["mean_confidence_max"]["value"]:
            decision = "cloud"; reasons.append("low_mean_conf")
        elif feats["object_density_raw"] > fc["object_density_max"]["value"]:
            decision = "cloud"; reasons.append("dense")

        # force_local (仅在未触发 force_cloud 时检查)
        if decision == "local":
            if feats["object_density_raw"] < fl["object_density_max"]["value"]:
                decision = "local"; reasons.append("sparse")
            elif feats["mean_confidence_raw"] > fl["mean_confidence_min"]["value"]:
                decision = "local"; reasons.append("high_conf")
            elif feats["laplacian_var_raw"] > fl["blur_laplacian_min"]["value"]:
                decision = "local"; reasons.append("sharp")
            else:
                reasons.append("default")

        if decision == "cloud":
            num_cloud += 1
            cloud_reasons.extend(reasons)
        else:
            num_local += 1

    total = num_cloud + num_local
    from collections import Counter
    reason_counts = dict(Counter(cloud_reasons))

    return {
        "total_images": total,
        "num_cloud": num_cloud,
        "num_local": num_local,
        "cloud_ratio": round(num_cloud / total, 4) if total > 0 else 0,
        "cloud_reason_breakdown": reason_counts,
    }


def main():
    args = parse_args()
    data_dir = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("动态路由阈值标定")
    print("=" * 70)
    print(f"数据集: {data_dir} / {args.split}")
    print(f"模型: {args.engine}")
    print(f"输出: {output_path}")
    print()

    # 1. 获取图片列表
    pairs = get_image_label_pairs(data_dir, args.split)
    if args.max_images > 0:
        pairs = pairs[:args.max_images]
    print(f"图片总数: {len(pairs)}")

    # 2. 初始化模型
    print("\n初始化 EdgeDetector ...")
    detector = EdgeDetector(
        weights=args.engine,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    print("初始化 ComplexityEvaluator ...")
    evaluator = ComplexityEvaluator()

    # 3. 提取特征
    print(f"\n提取特征中 ({len(pairs)} 张) ...")
    all_features = []
    t_start = time.time()

    for i, (img_path, label_path) in enumerate(pairs):
        try:
            feats, dets = extract_features(img_path, detector, evaluator)
            all_features.append(feats)
        except Exception as e:
            print(f"  [WARN] 跳过 {img_path.name}: {e}")
            continue

        if (i + 1) % 100 == 0 or i == len(pairs) - 1:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            print(f"  已处理 {i+1}/{len(pairs)} 张  ({fps:.1f} fps, 耗时 {elapsed:.1f}s)")

    print(f"\n成功提取: {len(all_features)} 张")

    # 4. 统计分布
    print("\n" + "=" * 70)
    print("13 维特征统计分布")
    print("=" * 70)

    raw_keys = [k for k in all_features[0].keys() if k.endswith("_raw")]
    stats = {}

    for key in raw_keys:
        values = [f[key] for f in all_features]
        s = compute_stats(values)
        stats[key] = s

        short_name = key.replace("_raw", "")
        print(f"\n  {short_name:<20} ({key})")
        print(f"    min={s['min']:>10.3f}  mean={s['mean']:>10.3f}  std={s['std']:>10.3f}  max={s['max']:>10.3f}")
        print(f"    P10={s['p10']:>10.3f}  P25={s['p25']:>10.3f}  P50={s['p50']:>10.3f}  P75={s['p75']:>10.3f}  P90={s['p90']:>10.3f}")

    # 5. 阈值建议
    print("\n" + "=" * 70)
    print("阈值建议")
    print("=" * 70)
    suggestions = suggest_thresholds(stats)

    print("\n  [force_cloud] 满足任一即回退云端:")
    for rule, info in suggestions["force_cloud"].items():
        print(f"    {rule:<30} = {info['value']:<10}  # {info['rationale']}")

    print("\n  [force_local] 满足任一即本地推理:")
    for rule, info in suggestions["force_local"].items():
        print(f"    {rule:<30} = {info['value']:<10}  # {info['rationale']}")

    # 6. 模拟路由
    print("\n" + "=" * 70)
    print("路由模拟（用建议阈值在 val 集上测试）")
    print("=" * 70)
    sim_result = simulate_routing(all_features, suggestions)
    print(f"\n  总样本:      {sim_result['total_images']}")
    print(f"  云端回退:    {sim_result['num_cloud']} ({sim_result['cloud_ratio']*100:.2f}%)")
    print(f"  本地推理:    {sim_result['num_local']} ({(1-sim_result['cloud_ratio'])*100:.2f}%)")
    print(f"\n  回退原因分布:")
    for reason, count in sorted(sim_result["cloud_reason_breakdown"].items(), key=lambda x: -x[1]):
        pct = count / sim_result['num_cloud'] * 100 if sim_result['num_cloud'] > 0 else 0
        print(f"    {reason:<20} {count:>5}  ({pct:>5.1f}%)")

    # 7. 保存结果
    result = {
        "dataset": str(data_dir),
        "split": args.split,
        "engine": args.engine,
        "total_images": len(pairs),
        "success_images": len(all_features),
        "feature_statistics": stats,
        "threshold_suggestions": suggestions,
        "simulation": sim_result,
        "recommended_config": {
            "routing": {
                "strategy": "rule_based",
                "rule_based": {
                        "force_cloud": {
                            k: v["value"] for k, v in suggestions["force_cloud"].items()
                        },
                        "force_local": {
                            k: v["value"] for k, v in suggestions["force_local"].items()
                        },
                },
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
