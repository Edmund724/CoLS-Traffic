"""
场景复杂度评估器 (Scene Complexity Evaluator)

同时提供：
  - 图像级特征（无模型开销）：熵值、光照、模糊度
  - 检测级特征（基于本地检测结果）：目标密度、置信度分布、类别多样性
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class ComplexityEvaluator:
    """
    场景复杂度评估器。

    使用方式：
        eval = ComplexityEvaluator()
        img_feats = eval.image_features(image)
        det_feats = eval.detection_features(detections, image_shape)
        feats = {**img_feats, **det_feats}
    """

    def __init__(self):
        # 归一化参数（基于经验值，后续可根据数据集统计更新）
        self.norm_params = {
            "entropy": {"min": 4.0, "max": 8.0},
            "brightness_mean": {"min": 0.0, "max": 255.0},
            "brightness_std": {"min": 0.0, "max": 100.0},
            "overexposure_ratio": {"min": 0.0, "max": 1.0},
            "underexposure_ratio": {"min": 0.0, "max": 1.0},
            "laplacian_var": {"min": 0.0, "max": 500.0},
            "contrast": {"min": 0.0, "max": 100.0},
            "object_density": {"min": 0.0, "max": 1.0},
            "object_count_norm": {"min": 0.0, "max": 50.0},
            "mean_confidence": {"min": 0.0, "max": 1.0},
            "low_conf_ratio": {"min": 0.0, "max": 1.0},
            "class_entropy": {"min": 0.0, "max": 3.0},
            "mean_box_size": {"min": 0.0, "max": 1.0},
        }

    # ── 图像级特征 ──────────────────────────────────────────────────────────

    def image_features(self, image: np.ndarray) -> dict[str, float]:
        """
        从原始 RGB 图像提取无模型特征。

        Args:
            image: [H, W, 3] uint8 RGB 图像

        Returns:
            dict: 包含 entropy, brightness_mean, brightness_std,
                  overexposure_ratio, underexposure_ratio,
                  laplacian_var, contrast 的原始值和归一化值
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        total_pixels = h * w

        # 1. 信息熵
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))

        # 2. 亮度统计
        brightness_mean = float(gray.mean())
        brightness_std = float(gray.std())

        # 3. 过曝 / 欠曝比例
        overexposed = np.sum(gray > 250) / total_pixels
        underexposed = np.sum(gray < 10) / total_pixels

        # 4. 模糊度（拉普拉斯方差）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())

        # 5. 对比度
        contrast = brightness_std / (brightness_mean + 1e-6) * 100.0

        raw = {
            "entropy": entropy,
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "overexposure_ratio": overexposed,
            "underexposure_ratio": underexposed,
            "laplacian_var": laplacian_var,
            "contrast": contrast,
        }

        return self._normalize(raw)

    # ── 检测级特征 ──────────────────────────────────────────────────────────

    def detection_features(
        self,
        detections: np.ndarray,
        image_shape: tuple[int, int],
    ) -> dict[str, float]:
        """
        从本地检测结果提取特征。

        Args:
            detections: [N, 6] = (x1, y1, x2, y2, conf, cls) 原始图像坐标
            image_shape: (H, W)

        Returns:
            dict: 包含 object_density, object_count_norm, mean_confidence,
                  low_conf_ratio, class_entropy, mean_box_size
        """
        h, w = image_shape
        img_area = h * w

        if len(detections) == 0:
            raw = {
                "object_density": 0.0,
                "object_count_norm": 0.0,
                "mean_confidence": 0.0,
                "low_conf_ratio": 0.0,
                "class_entropy": 0.0,
                "mean_box_size": 0.0,
            }
            return self._normalize(raw)

        boxes = detections[:, :4]   # xyxy
        confs = detections[:, 4]
        clses = detections[:, 5].astype(int)

        # 1. 目标密度 = 所有框面积占图像面积比例
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box_areas = np.clip(box_areas, 0, None)
        object_density = float(np.sum(box_areas) / img_area)
        object_density = min(object_density, 1.0)

        # 2. 归一化目标数量（假设最多50个目标）
        object_count_norm = float(len(detections))

        # 3. 平均置信度
        mean_confidence = float(np.mean(confs))

        # 4. 低置信度框比例（<0.5）
        low_conf_ratio = float(np.mean(confs < 0.5))

        # 5. 类别分布熵（衡量类别多样性）
        unique, counts = np.unique(clses, return_counts=True)
        probs = counts / counts.sum()
        class_entropy = float(-np.sum(probs * np.log2(probs + 1e-7)))

        # 6. 平均框大小（归一化到图像面积）
        mean_box_size = float(np.mean(box_areas / img_area))

        raw = {
            "object_density": object_density,
            "object_count_norm": object_count_norm,
            "mean_confidence": mean_confidence,
            "low_conf_ratio": low_conf_ratio,
            "class_entropy": class_entropy,
            "mean_box_size": mean_box_size,
        }

        return self._normalize(raw)

    # ── 归一化 ──────────────────────────────────────────────────────────────

    def _normalize(self, raw: dict[str, float]) -> dict[str, float]:
        """将原始特征值归一化到 [0, 1] 区间。"""
        out = {}
        for key, value in raw.items():
            if key in self.norm_params:
                p = self.norm_params[key]
                norm = (value - p["min"]) / (p["max"] - p["min"] + 1e-6)
                out[key] = float(np.clip(norm, 0.0, 1.0))
            else:
                out[key] = float(value)
            # 同时保留原始值（带 _raw 后缀）
            out[f"{key}_raw"] = float(value)
        return out

    # ── 便捷方法 ────────────────────────────────────────────────────────────

    def get_feature_vector(self, features: dict[str, float], keys: list[str] | None = None) -> np.ndarray:
        """
        将特征字典转为固定顺序的 numpy 向量（仅取归一化值，不含 _raw）。
        """
        if keys is None:
            keys = [
                "entropy", "brightness_mean", "brightness_std",
                "overexposure_ratio", "underexposure_ratio",
                "laplacian_var", "contrast",
                "object_density", "object_count_norm", "mean_confidence",
                "low_conf_ratio", "class_entropy", "mean_box_size",
            ]
        vec = [features.get(k, 0.0) for k in keys]
        return np.array(vec, dtype=np.float32)

    def explain(self, features: dict[str, float]) -> list[str]:
        """生成人类可读的特征解释（用于调试）。
        
        阈值基于 Jetson 实测 DAIR-V2X val 集 4464 张统计标定：
        - force_cloud: blur<31.4(P10), low_conf>0.50(P90), mean_conf<0.54(P10), dense>0.32(P90)
        - force_local: sparse<0.013(P25), mean_conf>0.763(P75), sharp>118.0(P75)
        """
        reasons = []
        if features.get("laplacian_var_raw", 100) < 31.4:
            reasons.append(f"图像模糊 (拉普拉斯方差={features['laplacian_var_raw']:.1f} < 31.4)")
        # 过曝在数据集中几乎不存在（max=0.007%），实际不触发
        # if features.get("overexposure_ratio_raw", 0) > 0.01:
        #     reasons.append(f"过曝严重 (比例={features['overexposure_ratio_raw']:.2%} > 0.01)")
        if features.get("underexposure_ratio_raw", 0) > 0.026:
            reasons.append(f"欠曝严重 (比例={features['underexposure_ratio_raw']:.2%} > 0.026)")
        if features.get("object_density_raw", 0) > 0.32:
            reasons.append(f"目标极度密集 (密度={features['object_density_raw']:.2%} > 32%)")
        if features.get("mean_confidence_raw", 1.0) < 0.54:
            reasons.append(f"平均置信度低 (均值={features['mean_confidence_raw']:.2f} < 0.54)")
        if features.get("low_conf_ratio_raw", 0) > 0.50:
            reasons.append(f"低置信度框过多 (比例={features['low_conf_ratio_raw']:.2%} > 50%)")
        # force_local 条件（用于调试输出）
        if features.get("object_density_raw", 1.0) < 0.013:
            reasons.append(f"场景空旷 (密度={features['object_density_raw']:.2%} < 1.3%)")
        if features.get("mean_confidence_raw", 0) > 0.763:
            reasons.append(f"模型高度确信 (均值={features['mean_confidence_raw']:.2f} > 0.763)")
        if features.get("laplacian_var_raw", 0) > 118.0:
            reasons.append(f"图像非常清晰 (拉普拉斯方差={features['laplacian_var_raw']:.1f} > 118.0)")
        return reasons
