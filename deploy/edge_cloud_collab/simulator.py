"""
云边协同仿真器 (Cloud-Edge Collaboration Simulator)

模拟并评测三种策略：
  1. Edge-Only   : 所有图像走本地边缘推理
  2. Cloud-Only  : 所有图像走云端大模型（不现实，仅用于上界参考）
  3. Dynamic     : 根据复杂度动态路由
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cloud_client import CloudClient
from .complexity_evaluator import ComplexityEvaluator
from .dynamic_router import DynamicRouter
from .edge_detector import EdgeDetector
from .utils import load_config, load_image


class CloudEdgeSimulator:
    """
    云边协同仿真器。

    Usage:
        sim = CloudEdgeSimulator.from_config()
        result = sim.run_single("image.jpg")
        stats = sim.run_batch(["img1.jpg", "img2.jpg", ...])
    """

    def __init__(
        self,
        edge_detector: EdgeDetector,
        complexity_evaluator: ComplexityEvaluator,
        router: DynamicRouter,
        cloud_client: CloudClient,
        network_cfg: dict | None = None,
    ):
        self.detector = edge_detector
        self.evaluator = complexity_evaluator
        self.router = router
        self.cloud = cloud_client
        self.network_cfg = network_cfg or {}

        # 延迟模拟参数
        self.edge_latency = {
            "mean": 25.0,
            "std": 3.0,
            "min": 15.0,
        }
        self.cloud_latency = {
            "mean": 300.0,
            "std": 50.0,
            "min": 200.0,
        }
        self.network = {
            "bandwidth_mbps": 10.0,
            "base_rtt_ms": 20.0,
            "jitter_ms": 5.0,
        }

    # ── 单帧仿真 ────────────────────────────────────────────────────────────

    def run_single(
        self,
        image: str | Path | np.ndarray,
        strategy: str = "dynamic",
    ) -> dict[str, Any]:
        """
        对单张图像跑完整流程。

        Args:
            image: 图像路径或 RGB 数组
            strategy: "edge_only" | "cloud_only" | "dynamic"

        Returns:
            dict: 包含检测框、延迟、路由决策、特征等完整信息
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            img_rgb = load_image(image)
            img_path = str(image)
        else:
            img_rgb = image
            img_path = "<ndarray>"

        h, w = img_rgb.shape[:2]
        record = {
            "image_path": img_path,
            "original_shape": (h, w),
            "strategy": strategy,
        }

        # ── Step 1: 图像级复杂度特征 ──────────────────────────────────────
        t0 = time.perf_counter()
        img_feats = self.evaluator.image_features(img_rgb)
        feat_time = (time.perf_counter() - t0) * 1000.0

        # ── Step 2: 本地边缘检测 ──────────────────────────────────────────
        edge_result = self.detector.predict(img_rgb)
        edge_dets = edge_result["detections"]
        edge_latency = edge_result["latency_ms"]

        # 模拟部署延迟：仅在 use_simulated_latency=True 时覆盖真实延迟
        # 真实 Jetson 部署时应设为 False，使用 TRT 实测值
        if getattr(self, "use_simulated_latency", True):
            if edge_latency < self.edge_latency["min"]:
                edge_latency = random.gauss(self.edge_latency["mean"], self.edge_latency["std"])
                edge_latency = max(self.edge_latency["min"], edge_latency)

        # ── Step 3: 检测级复杂度特征 ──────────────────────────────────────
        det_feats = self.evaluator.detection_features(edge_dets, (h, w))
        features = {**img_feats, **det_feats}

        # ── Step 4: 路由决策 ──────────────────────────────────────────────
        t_route0 = time.perf_counter()
        if strategy == "edge_only":
            decision = "local"
            route_info = {"reasons": ["策略强制: 仅边缘"], "rule": "edge_only"}
        elif strategy == "cloud_only":
            decision = "cloud"
            route_info = {"reasons": ["策略强制: 仅云端"], "rule": "cloud_only"}
        else:
            decision, route_info = self.router.decide(features)
        route_time = (time.perf_counter() - t_route0) * 1000.0

        record["decision"] = decision
        record["route_info"] = route_info
        record["features"] = features
        record["feature_extraction_ms"] = feat_time
        record["routing_decision_ms"] = route_time

        # ── Step 5: 执行推理 ──────────────────────────────────────────────
        if decision == "local":
            # 本地推理路径
            record["detections"] = edge_dets
            record["latency_ms"] = edge_latency
            record["edge_latency_ms"] = edge_latency
            record["network_latency_ms"] = 0.0
            record["cloud_latency_ms"] = 0.0
            record["cloud_success"] = None

        else:
            # 云端回退路径
            # 5a. 网络传输延迟（图像 + 检测结果上传）
            img_size_mb = (h * w * 3) / (1024 * 1024)  # RGB 原始大小（MB）
            # JPEG 压缩后大约 1/10
            img_size_mb /= 10.0
            bandwidth_mbps = self.network["bandwidth_mbps"]
            upload_ms = (img_size_mb * 8 / bandwidth_mbps) * 1000.0
            jitter = random.uniform(-self.network["jitter_ms"], self.network["jitter_ms"])
            network_latency = self.network["base_rtt_ms"] + upload_ms + jitter
            network_latency = max(5.0, network_latency)

            # 5b. 云端推理
            cloud_result = self.cloud.correct(img_rgb, edge_dets, {})
            cloud_latency = cloud_result["latency_ms"]
            cloud_dets = cloud_result["detections"]

            # 5c. 总延迟
            total_latency = edge_latency + network_latency + cloud_latency

            record["detections"] = cloud_dets
            record["latency_ms"] = total_latency
            record["edge_latency_ms"] = edge_latency
            record["network_latency_ms"] = network_latency
            record["cloud_latency_ms"] = cloud_latency
            record["cloud_success"] = cloud_result.get("success", True)
            record["cloud_mock"] = cloud_result.get("mock", False)

        return record

    # ── 批量仿真 ────────────────────────────────────────────────────────────

    def run_batch(
        self,
        images: list[str | Path | np.ndarray],
        strategy: str = "dynamic",
    ) -> dict[str, Any]:
        """
        批量仿真，返回统计报告。
        图片会先统一加载到内存，避免重复文件 I/O（模拟真实场景摄像头帧）。

        Returns:
            dict: 包含延迟分布、路由分布、成功率等统计信息
        """
        # 预加载所有图片到内存
        loaded = []
        for img in images:
            if isinstance(img, (str, Path)):
                loaded.append(load_image(img))
            else:
                loaded.append(img)

        records = []
        for img_rgb in loaded:
            try:
                rec = self.run_single(img_rgb, strategy=strategy)
                records.append(rec)
            except Exception as e:
                print(f"[Simulator] 处理失败: {e}")

        if len(records) == 0:
            return {"error": "无有效记录"}

        # 提取统计量
        latencies = [r["latency_ms"] for r in records]
        decisions = [r["decision"] for r in records]
        edge_latencies = [r["edge_latency_ms"] for r in records]
        cloud_latencies = [r.get("cloud_latency_ms", 0.0) for r in records]
        network_latencies = [r.get("network_latency_ms", 0.0) for r in records]

        num_cloud = sum(1 for d in decisions if d == "cloud")
        num_local = len(decisions) - num_cloud

        # 本地样本端到端延迟（含特征提取 + 路由判决）
        local_records = [r for r in records if r.get("decision") == "local"]
        if local_records:
            local_e2e = [
                r["latency_ms"] + r.get("feature_extraction_ms", 0.0) + r.get("routing_decision_ms", 0.0)
                for r in local_records
            ]
            local_e2e_stats = {
                "mean": float(np.mean(local_e2e)),
                "p50": float(np.percentile(local_e2e, 50)),
                "p90": float(np.percentile(local_e2e, 90)),
            }
        else:
            local_e2e_stats = {"mean": 0.0, "p50": 0.0, "p90": 0.0}

        report = {
            "strategy": strategy,
            "total_images": len(records),
            "num_local": num_local,
            "num_cloud": num_cloud,
            "cloud_ratio": num_cloud / len(records) if records else 0.0,
            "latency_ms": {
                "mean": float(np.mean(latencies)),
                "median": float(np.median(latencies)),
                "p50": float(np.percentile(latencies, 50)),
                "p90": float(np.percentile(latencies, 90)),
                "p99": float(np.percentile(latencies, 99)),
                "max": float(np.max(latencies)),
                "min": float(np.min(latencies)),
            },
            "edge_latency_ms": {
                "mean": float(np.mean(edge_latencies)),
            },
            "cloud_latency_ms": {
                "mean": float(np.mean(cloud_vals)) if (cloud_vals := [x for x in cloud_latencies if x > 0]) else 0.0,
            },
            "network_latency_ms": {
                "mean": float(np.mean(net_vals)) if (net_vals := [x for x in network_latencies if x > 0]) else 0.0,
            },
            "local_e2e_ms": local_e2e_stats,
            "records": records,
        }

        # 路由原因统计
        reason_counts = {}
        for r in records:
            rule = r["route_info"].get("rule", "unknown")
            reason_counts[rule] = reason_counts.get(rule, 0) + 1
        report["route_distribution"] = reason_counts

        return report

    # ── 对比实验 ────────────────────────────────────────────────────────────

    def compare_strategies(self, images: list[str | Path]) -> dict[str, Any]:
        """
        对同一批图像运行动态路由策略，输出延迟指标。
        """
        print(f"[Simulator] 运行动态路由策略: {len(images)} 张图像")

        report = self.run_batch(images, strategy="dynamic")
        results = {"dynamic": report}

        # 打印动态路由策略延迟报告
        print("\n" + "=" * 70)
        print("  云边协同动态路由策略延迟报告")
        print("=" * 70)

        rep = report
        if "error" not in rep:
            print(f"  总样本数          : {rep['total_images']}")
            print(f"  本地推理样本      : {rep['num_local']} ({rep['num_local']/rep['total_images']:.1%})")
            print(f"  云端回退样本      : {rep['num_cloud']} ({rep['cloud_ratio']:.1%})")
            print("-" * 70)

            # 1. 本地路由延迟（仅本地样本）
            if rep['num_local'] > 0:
                print("  本地路由延迟（仅本地样本）:")
                print(f"    均值 : {rep['local_e2e_ms']['mean']:>8.1f} ms  (边缘推理 + 特征提取 + 路由判决)")
                print(f"    P90  : {rep['local_e2e_ms']['p90']:>8.1f} ms")
                print()

            # 2. 混合策略延迟（全部样本的平均）
            lat = rep['latency_ms']
            print("  混合策略延迟（全部样本平均）:")
            print(f"    均值 : {lat['mean']:>8.1f} ms  (本地快 + 云端慢 加权平均)")
            print(f"    P90  : {lat['p90']:>8.1f} ms")
            print()

            # 3. 云端回退延迟（仅云端样本）
            if rep['num_cloud'] > 0:
                cloud_mean = rep['cloud_latency_ms']['mean']
                net_mean = rep['network_latency_ms']['mean']
                print("  云端回退延迟（仅云端样本）:")
                print(f"    边缘预处理 : {rep['edge_latency_ms']['mean']:>8.1f} ms")
                print(f"    网络传输   : {net_mean:>8.1f} ms")
                print(f"    云端 VLM   : {cloud_mean:>8.1f} ms")
                print(f"    云端总延迟 : {rep['edge_latency_ms']['mean'] + net_mean + cloud_mean:>8.1f} ms")

            print("-" * 70)

        # 路由分布
        if "route_distribution" in rep:
            print("  路由决策分布:")
            for rule, count in rep["route_distribution"].items():
                print(f"    - {rule}: {count} ({count / rep['total_images']:.1%})")

        print("=" * 70)
        return results

    # ── 便捷工厂 ────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "configs/edge_cloud.yaml") -> "CloudEdgeSimulator":
        """从配置文件一键实例化仿真器。"""
        cfg = load_config(config_path)

        edge = EdgeDetector.from_config(cfg["edge_detector"])
        evaluator = ComplexityEvaluator()
        router = DynamicRouter.from_config(cfg["routing"])
        cloud = CloudClient.from_config(cfg["cloud"], mock=not cfg["simulator"].get("use_real_api", False))

        sim = cls(
            edge_detector=edge,
            complexity_evaluator=evaluator,
            router=router,
            cloud_client=cloud,
        )

        # 加载延迟模拟参数
        sim.edge_latency = cfg["simulator"].get("edge_latency_ms", sim.edge_latency)
        sim.cloud_latency = cfg["simulator"].get("cloud_inference_ms", sim.cloud_latency)
        sim.network = cfg["simulator"].get("network", sim.network)
        sim.use_simulated_latency = cfg["simulator"].get("use_simulated_latency", True)

        return sim
