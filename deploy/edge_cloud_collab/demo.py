"""
云边协同动态路由系统演示脚本

Usage:
    # 单图演示（打印全流程）
    python -m edge_cloud_collab.demo --image datasets/dair_v2x_yolo/images/train/000000.jpg

    # 批量对比实验（默认取 val 集前 N 张）
    python -m edge_cloud_collab.demo --batch 20

    # 指定配置文件
    python -m edge_cloud_collab.demo --config configs/edge_cloud.yaml --batch 10
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from edge_cloud_collab import (
    CloudClient,
    CloudEdgeSimulator,
    ComplexityEvaluator,
    DynamicRouter,
    EdgeDetector,
    load_config,
)
from edge_cloud_collab.utils import draw_detections, load_image


def demo_single(image_path: str, config_path: str = "configs/edge_cloud.yaml"):
    """单图全流程演示。"""
    print("=" * 80)
    print(f" 单图演示: {image_path}")
    print("=" * 80)

    cfg = load_config(config_path)

    # 初始化模块
    print("\n[1/5] 初始化模块...")
    detector = EdgeDetector.from_config(cfg["edge_detector"])
    detector.warmup(n=5)   # 预热，消除冷启动
    evaluator = ComplexityEvaluator()
    router = DynamicRouter.from_config(cfg["routing"])
    cloud = CloudClient.from_config(cfg["cloud"], mock=True)

    # 加载图像
    print("[2/5] 加载图像...")
    img_rgb = load_image(image_path)
    h, w = img_rgb.shape[:2]
    print(f"      图像尺寸: {w}x{h}")

    # 本地边缘检测
    print("[3/5] 本地边缘检测...")
    edge_result = detector.predict(img_rgb)
    edge_dets = edge_result["detections"]
    print(f"      检测到 {len(edge_dets)} 个目标, 耗时 {edge_result['latency_ms']:.1f} ms")

    # 复杂度评估
    print("[4/5] 场景复杂度评估...")
    img_feats = evaluator.image_features(img_rgb)
    det_feats = evaluator.detection_features(edge_dets, (h, w))
    features = {**img_feats, **det_feats}

    print("      图像级特征:")
    for k in ["entropy", "brightness_mean", "laplacian_var", "overexposure_ratio", "underexposure_ratio"]:
        raw_key = f"{k}_raw"
        if raw_key in features:
            print(f"        - {k}: {features[raw_key]:.3f} (归一化: {features[k]:.3f})")

    print("      检测级特征:")
    for k in ["object_density", "mean_confidence", "low_conf_ratio", "class_entropy"]:
        raw_key = f"{k}_raw"
        if raw_key in features:
            print(f"        - {k}: {features[raw_key]:.3f} (归一化: {features[k]:.3f})")

    explanations = evaluator.explain(features)
    if explanations:
        print("      特征解释:")
        for exp in explanations:
            print(f"        ⚠️ {exp}")

    # 路由决策
    print("[5/5] 动态路由决策...")
    decision, route_info = router.decide(features)
    print(f"      决策结果: {'🟢 本地推理' if decision == 'local' else '🔴 云端回退'}")
    print(f"      决策理由: {route_info['reasons']}")

    # 执行决策并展示结果
    if decision == "local":
        final_dets = edge_dets
        print(f"\n  ✅ 最终输出: {len(final_dets)} 个目标 (本地推理)")
    else:
        print("\n  调用云端大模型协同纠错...")
        names = cfg["dataset"].get("names", {})
        # 转换 names 为 {int: str}
        names = {int(k): v for k, v in names.items()} if names else {}
        cloud_result = cloud.correct(img_rgb, edge_dets, names)
        final_dets = cloud_result["detections"]
        print(f"      云端耗时: {cloud_result['latency_ms']:.1f} ms")
        print(f"      云端成功: {cloud_result.get('success', False)}")
        print(f"\n  ✅ 最终输出: {len(final_dets)} 个目标 (云端精修)")

    # 保存可视化结果
    vis = draw_detections(img_rgb, final_dets, names=cfg["dataset"].get("names", {}))
    out_path = "edge_cloud_collab_demo_result.jpg"
    import cv2
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\n  📷 可视化结果已保存: {out_path}")


def demo_batch(num_images: int = 20, config_path: str = "configs/edge_cloud.yaml"):
    """批量对比实验。"""
    print("=" * 80)
    print(f" 批量对比实验: 随机抽取 {num_images} 张图像")
    print("=" * 80)

    cfg = load_config(config_path)
    data_yaml = cfg["dataset"]["data_yaml"]

    # 查找图像
    data_path = Path(data_yaml).parent
    val_dir = data_path / "images" / "val"
    train_dir = data_path / "images" / "train"

    image_dir = val_dir if val_dir.exists() else train_dir
    if not image_dir.exists():
        print(f"错误: 找不到图像目录 {image_dir}")
        return

    all_images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if len(all_images) == 0:
        print(f"错误: 目录 {image_dir} 中没有图像")
        return

    # 随机采样
    if len(all_images) > num_images:
        selected = random.sample(all_images, num_images)
    else:
        selected = all_images

    print(f"\n 数据集: {data_yaml}")
    print(f" 图像目录: {image_dir}")
    print(f" 可用图像: {len(all_images)}, 本次抽取: {len(selected)}")

    # 初始化仿真器
    print("\n 初始化仿真器...")
    sim = CloudEdgeSimulator.from_config(config_path)

    # 运行对比实验
    results = sim.compare_strategies([str(p) for p in selected])

    # 详细分析 dynamic 策略下的典型样本
    dyn = results.get("dynamic", {})
    records = dyn.get("records", [])

    print("\n" + "-" * 80)
    print(" 典型样本分析 (Dynamic 策略):")
    print("-" * 80)

    # 找几个云端回退的样本
    cloud_samples = [r for r in records if r["decision"] == "cloud"]
    if cloud_samples:
        sample = cloud_samples[0]
        print(f"\n  [云端回退样本] {sample['image_path']}")
        print(f"    延迟: {sample['latency_ms']:.1f} ms")
        print(f"    边缘: {sample['edge_latency_ms']:.1f} ms | "
              f"网络: {sample.get('network_latency_ms', 0):.1f} ms | "
              f"云端: {sample.get('cloud_latency_ms', 0):.1f} ms")
        print(f"    理由: {sample['route_info']['reasons']}")
        explanations = ComplexityEvaluator().explain(sample['features'])
        for exp in explanations:
            print(f"    ⚠️ {exp}")

    # 找几个本地推理的样本
    local_samples = [r for r in records if r["decision"] == "local"]
    if local_samples:
        sample = local_samples[0]
        print(f"\n  [本地推理样本] {sample['image_path']}")
        print(f"    延迟: {sample['latency_ms']:.1f} ms")
        print(f"    理由: {sample['route_info']['reasons']}")

    print("\n" + "=" * 80)
    print(" 演示完成")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="云边协同动态路由演示")
    parser.add_argument("--image", type=str, default=None, help="单图路径")
    parser.add_argument("--batch", type=int, default=None, help="批量实验图像数")
    parser.add_argument("--config", type=str, default="configs/edge_cloud.yaml", help="配置文件路径")
    args = parser.parse_args()

    if args.image:
        demo_single(args.image, args.config)
    elif args.batch:
        demo_batch(args.batch, args.config)
    else:
        # 默认：先跑单图，再跑小批量
        print("未指定 --image 或 --batch，运行默认演示...")
        print("\n" + "=" * 80)
        print(" 默认演示 1: 单图全流程")
        print("=" * 80)

        cfg = load_config(args.config)
        data_path = Path(cfg["dataset"]["data_yaml"]).parent
        train_dir = data_path / "images" / "train"
        if train_dir.exists():
            sample_imgs = sorted(list(train_dir.glob("*.jpg")))
            if sample_imgs:
                demo_single(str(sample_imgs[0]), args.config)

        print("\n" + "=" * 80)
        print(" 默认演示 2: 批量对比 (10张)")
        print("=" * 80)
        demo_batch(10, args.config)


if __name__ == "__main__":
    main()
