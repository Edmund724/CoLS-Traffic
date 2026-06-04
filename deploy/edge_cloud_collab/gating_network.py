"""
场景复杂度门控网络 (Gating Network)

功能：
  1. 输入：场景复杂度特征向量（来自 ComplexityEvaluator）
  2. 输出：云端回退概率 P(cloud) ∈ [0, 1]
  3. 支持 rule_based（规则基线）、mlp（轻量神经网络）两种模式
  4. 支持离线训练（用标注的本地/云端决策标签训练 MLP）

设计目标：
  - 零训练开销的规则基线用于快速部署
  - 可训练的 MLP 用于后续根据实际数据优化路由决策
  - 特征可解释：输出各特征对决策的贡献度
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class GatingNetwork(nn.Module):
    """
    轻量 MLP 门控网络。

    架构: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Sigmoid
    输出: P(cloud) ∈ [0, 1]
    """

    def __init__(
        self,
        in_dim: int = 13,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, D] → Returns: prob [B, 1]"""
        return self.net(x)

    def predict_prob(self, features_vec: np.ndarray) -> float:
        """输入 numpy 特征向量，返回 P(cloud)。"""
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(features_vec).float().unsqueeze(0)  # [1, D]
            prob = self(x).item()
        return prob

    def decide(
        self,
        features_vec: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[str, float, dict[str, Any]]:
        """
        做出路由决策。

        Returns:
            (decision, prob, info)
            decision: "local" | "cloud"
            prob: P(cloud)
            info: 包含决策理由的字典
        """
        prob = self.predict_prob(features_vec)
        decision = "cloud" if prob > threshold else "local"
        info = {
            "reasons": [f"MLP Gating: P(cloud)={prob:.3f}"],
            "prob_cloud": prob,
            "threshold": threshold,
            "rule": "mlp",
        }
        return decision, prob, info

    def feature_importance(
        self,
        features_vec: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        计算输入特征的敏感度（数值梯度近似）。
        用于解释模型为什么做出这个决策。
        """
        self.eval()
        x = torch.from_numpy(features_vec).float().unsqueeze(0)
        x.requires_grad = True

        prob = self(x)
        prob.backward()

        grads = x.grad.squeeze(0).abs().cpu().numpy()
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(grads))]

        return {name: float(g) for name, g in zip(feature_names, grads)}

    def save(self, path: str | Path):
        """保存模型权重。"""
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path):
        """加载模型权重。"""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()


# ── 训练接口 ────────────────────────────────────────────────────────────────

def train_gating_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.2,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
) -> tuple[GatingNetwork, dict[str, list[float]]]:
    """
    离线训练门控网络。

    Args:
        X_train: [N, D] 特征矩阵（归一化后的复杂度特征）
        y_train: [N] 标签，0=local, 1=cloud
        X_val, y_val: 验证集（可选）
        hidden_dims: MLP 隐藏层维度
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批次大小
        device: "cpu" | "cuda"

    Returns:
        (model, history)
        history: {"train_loss": [...], "val_acc": [...]}
    """
    in_dim = X_train.shape[1]
    model = GatingNetwork(in_dim=in_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.to(device)

    # 数据加载器
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float().unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(train_ds)
        history["train_loss"].append(epoch_loss)

        # 验证
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                x_val = torch.from_numpy(X_val).float().to(device)
                pred_val = model(x_val).squeeze().cpu().numpy()
                pred_labels = (pred_val > 0.5).astype(int)
                acc = np.mean(pred_labels == y_val)
                history["val_acc"].append(float(acc))

        if (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch + 1}/{epochs}, loss={epoch_loss:.4f}"
            if "val_acc" in history and history["val_acc"]:
                msg += f", val_acc={history['val_acc'][-1]:.3f}"
            print(f"[GatingNetwork] {msg}")

    return model, history


# ── 便捷工厂 ────────────────────────────────────────────────────────────────

def build_gating_network_from_config(cfg: dict) -> GatingNetwork | None:
    """从配置字典构建门控网络（支持加载预训练权重）。"""
    mlp_cfg = cfg.get("mlp", {})
    ckpt = mlp_cfg.get("checkpoint")
    if not ckpt:
        return None

    model = GatingNetwork(
        in_dim=mlp_cfg.get("in_dim", 13),
        hidden_dims=mlp_cfg.get("hidden_dims", [64, 32]),
        dropout=mlp_cfg.get("dropout", 0.2),
    )
    model.load(ckpt)
    print(f"[GatingNetwork] 加载预训练权重: {ckpt}")
    return model
