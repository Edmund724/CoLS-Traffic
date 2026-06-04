"""
动态路由决策模块 (Dynamic Router)

支持三种策略：
  1. rule_based  — 基于阈值的规则基线（零训练开销，可解释性强）
  2. mlp         — 轻量 MLP 门控网络（预留，后续可用离线数据训练）
  3. random      — 随机决策（用于消融实验 baseline）
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class DynamicRouter:
    """
    动态路由决策器。

    决策输入：场景复杂度特征字典（来自 ComplexityEvaluator）
    决策输出："local" 或 "cloud"
    """

    def __init__(self, strategy: str = "rule_based", config: dict | None = None):
        """
        Args:
            strategy: "rule_based" | "mlp" | "random"
            config: 对应策略的配置字典
        """
        self.strategy = strategy
        self.config = config or {}
        self._mlp_model: nn.Module | None = None

        if strategy == "mlp":
            self._build_mlp()

    # ── 公共接口 ────────────────────────────────────────────────────────────

    def decide(self, features: dict[str, float]) -> tuple[str, dict[str, Any]]:
        """
        根据场景复杂度特征做出路由决策。

        Returns:
            (decision, info)
            decision: "local" | "cloud"
            info: 包含决策理由的字典
        """
        if self.strategy == "rule_based":
            return self._rule_based(features)
        elif self.strategy == "mlp":
            return self._mlp_decide(features)
        elif self.strategy == "random":
            return self._random(features)
        else:
            raise ValueError(f"未知路由策略: {self.strategy}")

    # ── 规则基线 ────────────────────────────────────────────────────────────

    def _rule_based(self, features: dict[str, float]) -> tuple[str, dict[str, Any]]:
        """
        基于阈值的规则策略。

        优先级：
          1. 强制云端条件（任一满足 → cloud）
          2. 强制本地条件（任一满足 → local）
          3. 默认 → local（边缘优先，节省带宽和成本）
        """
        cfg = self.config.get("rule_based", {})
        force_cloud = cfg.get("force_cloud", {})
        force_local = cfg.get("force_local", {})

        reasons = []

        # ── 强制云端 ──────────────────────────────────────────────────────
        if "blur_laplacian_max" in force_cloud:
            val = features.get("laplacian_var_raw", 999)
            th = force_cloud["blur_laplacian_max"]
            if val < th:
                reasons.append(f"强制云端: 图像模糊 (拉普拉斯={val:.1f} < {th})")

        if "overexposure_min" in force_cloud:
            val = features.get("overexposure_ratio_raw", 0)
            th = force_cloud["overexposure_min"]
            if val > th:
                reasons.append(f"强制云端: 过曝严重 (比例={val:.2%} > {th})")

        if "underexposure_min" in force_cloud:
            val = features.get("underexposure_ratio_raw", 0)
            th = force_cloud["underexposure_min"]
            if val > th:
                reasons.append(f"强制云端: 欠曝严重 (比例={val:.2%} > {th})")

        if "low_conf_ratio_min" in force_cloud:
            val = features.get("low_conf_ratio_raw", 0)
            th = force_cloud["low_conf_ratio_min"]
            if val > th:
                reasons.append(f"强制云端: 低置信度框过多 (比例={val:.2%} > {th})")

        if "mean_confidence_max" in force_cloud:
            val = features.get("mean_confidence_raw", 1.0)
            obj_count = features.get("object_count_norm_raw", 0)
            th = force_cloud["mean_confidence_max"]
            # 仅当检测到目标时才检查平均置信度（空场景置信度为0是正常情况）
            if obj_count > 0 and val < th:
                reasons.append(f"强制云端: 平均置信度低 (均值={val:.2f} < {th})")

        if "object_density_max" in force_cloud:
            val = features.get("object_density_raw", 0)
            th = force_cloud["object_density_max"]
            if val > th:
                reasons.append(f"强制云端: 目标极度密集 (密度={val:.2%} > {th})")

        if reasons:
            return "cloud", {"reasons": reasons, "rule": "force_cloud"}

        # ── 强制本地 ──────────────────────────────────────────────────────
        local_reasons = []

        if "object_density_max" in force_local:
            val = features.get("object_density_raw", 1.0)
            th = force_local["object_density_max"]
            if val < th:
                local_reasons.append(f"强制本地: 场景空旷 (密度={val:.2%} < {th})")

        if "mean_confidence_min" in force_local:
            val = features.get("mean_confidence_raw", 0)
            th = force_local["mean_confidence_min"]
            if val > th:
                local_reasons.append(f"强制本地: 模型高度确信 (均值={val:.2f} > {th})")

        if "blur_laplacian_min" in force_local:
            val = features.get("laplacian_var_raw", 0)
            th = force_local["blur_laplacian_min"]
            if val > th:
                local_reasons.append(f"强制本地: 图像非常清晰 (拉普拉斯={val:.1f} > {th})")

        if local_reasons:
            return "local", {"reasons": local_reasons, "rule": "force_local"}

        # ── 默认本地 ──────────────────────────────────────────────────────
        return "local", {"reasons": ["默认策略: 边缘优先"], "rule": "default_local"}

    # ── MLP 门控网络 ────────────────────────────────────────────────────────

    def _build_mlp(self):
        """构建轻量 MLP 门控网络。"""
        cfg = self.config.get("mlp", {})
        in_dim = cfg.get("in_dim", 13)
        hidden = cfg.get("hidden_dims", [64, 32])
        dropout = cfg.get("dropout", 0.2)

        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        layers.append(nn.Sigmoid())

        self._mlp_model = nn.Sequential(*layers)
        self._mlp_model.eval()

        # 尝试加载预训练权重
        ckpt = cfg.get("checkpoint")
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            self._mlp_model.load_state_dict(state)
            print(f"[DynamicRouter] 加载 MLP 权重: {ckpt}")

    def _mlp_decide(self, features: dict[str, float]) -> tuple[str, dict[str, Any]]:
        """使用 MLP 门控网络做决策。"""
        if self._mlp_model is None:
            raise RuntimeError("MLP 模型未初始化")

        # 构建特征向量（与训练时顺序一致）
        from .complexity_evaluator import ComplexityEvaluator
        vec = ComplexityEvaluator().get_feature_vector(features)
        x = torch.from_numpy(vec).unsqueeze(0).float()  # [1, D]

        with torch.no_grad():
            prob = self._mlp_model(x).item()  # P(cloud)

        decision = "cloud" if prob > 0.5 else "local"
        return decision, {
            "reasons": [f"MLP 门控: P(cloud)={prob:.3f}"],
            "prob_cloud": prob,
            "rule": "mlp",
        }

    # ── 随机策略（消融 baseline） ───────────────────────────────────────────

    def _random(self, features: dict[str, float]) -> tuple[str, dict[str, Any]]:
        """随机决策，用于消融实验。"""
        decision = "cloud" if random.random() > 0.5 else "local"
        return decision, {"reasons": ["随机策略"], "rule": "random"}

    # ── 便捷方法 ────────────────────────────────────────────────────────────

    @staticmethod
    def from_config(cfg: dict) -> "DynamicRouter":
        """从配置字典实例化路由决策器。"""
        strategy = cfg.get("strategy", "rule_based")
        return DynamicRouter(strategy=strategy, config=cfg)
