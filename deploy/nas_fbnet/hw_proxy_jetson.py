"""
Jetson TRT 硬件代理：
  - 优先加载 nas_hw_mlp_per_target 产出的单目标 MLP
  - 兼容旧的 nas_hw_xgboost 产物

将 arch_config + 参数量映射为与 trt_benchmark_results 一致的特征行，并预测延迟/功耗。
"""
from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

from nas_fbnet.config_hw import NUM_STAGES


def _build_feature_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
    encoders: dict[str, LabelEncoder] | None,
    *,
    fit_encoders: bool,
) -> tuple[np.ndarray, list[str], dict[str, LabelEncoder]]:
    if numeric_cols:
        X_num = df[numeric_cols].to_numpy(dtype=np.float64)
    else:
        X_num = np.empty((len(df), 0), dtype=np.float64)
    encoders = encoders if encoders is not None else {}
    cat_blocks = []
    feature_names = list(numeric_cols)
    for c in cat_cols:
        if c not in df.columns:
            raise KeyError(f"缺少列: {c}")
        s = df[c].astype(str).fillna("__nan__")
        if fit_encoders:
            le = LabelEncoder()
            cat_blocks.append(le.fit_transform(s).reshape(-1, 1).astype(np.float32))
            encoders[c] = le
        else:
            le = encoders[c]
            known = set(le.classes_)
            s_safe = s.where(s.isin(known), le.classes_[0])
            cat_blocks.append(le.transform(s_safe).reshape(-1, 1).astype(np.float32))
        feature_names.append(c + "_le")
    blocks = [X_num.astype(np.float32)] + cat_blocks
    X = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
    return X, feature_names, encoders


def _norm(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class _NasHwMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JetsonHWProxy:
    """一次加载，多次 predict。"""

    def __init__(self, model_dir: str):
        self.model_dir = os.path.abspath(model_dir)
        self.device = torch.device("cpu")
        self.backend = ""
        self.targets: list[str] = []
        self.meta: dict[str, Any] = {}
        self.numeric_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []

        if self._can_load_mlp_per_target():
            self._load_mlp_per_target()
        elif self._can_load_mlp_multi_output():
            self._load_mlp_multi_output()
        else:
            self._load_xgboost()

    def _can_load_mlp_per_target(self) -> bool:
        meta_path = os.path.join(self.model_dir, "nas_hw_mlp_per_target_meta.json")
        return os.path.isfile(meta_path)

    def _can_load_mlp_multi_output(self) -> bool:
        meta_path = os.path.join(self.model_dir, "nas_hw_mlp_meta.json")
        ckpt_path = os.path.join(self.model_dir, "mlp_model.pt")
        pre_path = os.path.join(self.model_dir, "preprocess_mlp.joblib")
        return os.path.isfile(meta_path) and os.path.isfile(ckpt_path) and os.path.isfile(pre_path)

    def _load_common_blob(self, blob: dict[str, Any]) -> None:
        self.encoders = dict(blob["encoders"])
        self.numeric_cols = list(blob["numeric_cols"])
        self.cat_cols = list(blob["cat_cols"])
        self.feature_names = list(blob.get("feature_names") or blob.get("feature_names_after_encode") or [])

    def _load_mlp_per_target(self) -> None:
        self.backend = "mlp_per_target"
        meta_path = os.path.join(self.model_dir, "nas_hw_mlp_per_target_meta.json")
        shared_pre_path = os.path.join(self.model_dir, "preprocess_mlp_shared.joblib")
        if not os.path.isfile(shared_pre_path):
            raise FileNotFoundError(
                f"缺少 {shared_pre_path}，请先运行 hw_prediction/nas_hw_mlp_per_target.py"
            )
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        shared_blob = joblib.load(shared_pre_path)
        self._load_common_blob(shared_blob)
        self.targets = list(self.meta.get("targets") or [])
        if not self.targets:
            raise FileNotFoundError(f"{meta_path} 中 targets 为空")

        self.mlp_models: dict[str, nn.Module] = {}
        self.scaler_x_by_target: dict[str, StandardScaler] = {}
        self.scaler_y_by_target: dict[str, StandardScaler] = {}

        for name in self.targets:
            ckpt_path = os.path.join(self.model_dir, f"mlp_{name}.pt")
            pre_path = os.path.join(self.model_dir, f"preprocess_mlp_{name}.joblib")
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"缺少 MLP 模型: {ckpt_path}")
            if not os.path.isfile(pre_path):
                raise FileNotFoundError(f"缺少 MLP 预处理: {pre_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")
            model = _NasHwMLP(
                int(ckpt["in_dim"]),
                int(ckpt["out_dim"]),
                tuple(ckpt["hidden"]),
                float(ckpt["dropout"]),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.mlp_models[name] = model

            blob = joblib.load(pre_path)
            self.scaler_x_by_target[name] = blob["scaler_x"]
            self.scaler_y_by_target[name] = blob["scaler_y"]
            if not self.feature_names:
                self.feature_names = list(blob.get("feature_names") or [])

    def _load_mlp_multi_output(self) -> None:
        self.backend = "mlp_multi_output"
        meta_path = os.path.join(self.model_dir, "nas_hw_mlp_meta.json")
        pre_path = os.path.join(self.model_dir, "preprocess_mlp.joblib")
        ckpt_path = os.path.join(self.model_dir, "mlp_model.pt")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        blob = joblib.load(pre_path)
        self._load_common_blob(blob)
        self.targets = list(self.meta.get("targets") or [])
        if not self.targets:
            raise FileNotFoundError(f"{meta_path} 中 targets 为空")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = _NasHwMLP(
            int(ckpt["in_dim"]),
            int(ckpt["out_dim"]),
            tuple(ckpt["hidden"]),
            float(ckpt["dropout"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.mlp_model = model
        self.scaler_x = blob["scaler_x"]
        self.scaler_y = blob["scaler_y"]

    def _load_xgboost(self) -> None:
        self.backend = "xgboost"
        pre_path = os.path.join(self.model_dir, "preprocess.joblib")
        meta_path = os.path.join(self.model_dir, "nas_hw_xgb_meta.json")
        if not os.path.isfile(pre_path):
            raise FileNotFoundError(
                f"缺少 {pre_path}，请先运行 hw_prediction/nas_hw_mlp_per_target.py 或 hw_prediction/nas_hw_xgboost.py"
            )
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"缺少 {meta_path}")

        blob = joblib.load(pre_path)
        self.encoders = dict(blob["encoders"])
        self.numeric_cols = list(blob["numeric_cols"])
        self.cat_cols = list(blob["cat_cols"])

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        stems = sorted(
            fn[4:-5]
            for fn in os.listdir(self.model_dir)
            if fn.startswith("xgb_") and fn.endswith(".json")
        )
        meta_targets = list(self.meta.get("targets") or [])
        if stems:
            self.targets = stems
        else:
            self.targets = meta_targets
            if not self.targets:
                raise FileNotFoundError(
                    f"{self.model_dir} 下无 xgb_*.json 且 meta.targets 为空"
                )

        self.boosters: dict[str, xgb.Booster] = {}
        for name in self.targets:
            safe = name.replace("/", "_").replace(" ", "_")
            p = os.path.join(self.model_dir, f"xgb_{safe}.json")
            if not os.path.isfile(p):
                raise FileNotFoundError(f"缺少 XGBoost 模型: {p}")
            booster = xgb.Booster()
            booster.load_model(p)
            self.boosters[name] = booster

    def arch_config_to_row(self, arch_config: dict[str, Any], total_params: int) -> pd.DataFrame:
        depths = list(arch_config["depths"])
        if len(depths) != NUM_STAGES:
            raise ValueError(f"depths 长度应为 NUM_STAGES={NUM_STAGES}，收到 {len(depths)}")
        params_m = float(total_params) / 1e6
        depth_keys = [f"depth_s{i + 1}" for i in range(NUM_STAGES)]
        row = {
            "ks": int(arch_config["kernel_size"]),
            "expand": int(arch_config["expand_ratio"]),
            "width": float(arch_config.get("width_multiplier", 1.0)),
            **{depth_keys[i]: int(depths[i]) for i in range(NUM_STAGES)},
            "params_m": params_m,
            "infer_mode": str(arch_config.get("infer_mode", "")),
            "precision": str(arch_config.get("quant_precision", "")),
        }
        return pd.DataFrame([row])

    def _build_features(self, arch_config: dict[str, Any], total_params: int) -> tuple[np.ndarray, list[str]]:
        df = self.arch_config_to_row(arch_config, total_params)
        X, feat_names, _ = _build_feature_matrix(
            df,
            self.numeric_cols,
            self.cat_cols,
            self.encoders,
            fit_encoders=False,
        )
        return X, feat_names

    def _predict_raw_xgb(self, X: np.ndarray, feat_names: list[str]) -> dict[str, float]:
        meta_names = self.meta.get("feature_names_after_encode")
        if meta_names and list(meta_names) != list(feat_names):
            raise ValueError(
                f"特征名与 nas_hw_xgb_meta.json 不一致: 当前 {feat_names!r} vs meta {list(meta_names)!r}"
            )
        dm = xgb.DMatrix(X, feature_names=list(feat_names))
        out: dict[str, float] = {}
        for name in self.targets:
            out[name] = float(self.boosters[name].predict(dm)[0])
        return out

    def _predict_raw_mlp_per_target(self, X: np.ndarray, feat_names: list[str]) -> dict[str, float]:
        if self.feature_names and list(self.feature_names) != list(feat_names):
            raise ValueError(
                f"特征名与 MLP 训练产物不一致: 当前 {feat_names!r} vs 训练 {list(self.feature_names)!r}"
            )
        out: dict[str, float] = {}
        for name in self.targets:
            scaler_x = self.scaler_x_by_target[name]
            scaler_y = self.scaler_y_by_target[name]
            model = self.mlp_models[name]
            X_s = scaler_x.transform(X).astype(np.float32)
            with torch.no_grad():
                pred_s = model(torch.from_numpy(X_s)).cpu().numpy().reshape(-1, 1)
            pred = scaler_y.inverse_transform(pred_s).reshape(-1)
            out[name] = float(pred[0])
        return out

    def _predict_raw_mlp_multi_output(self, X: np.ndarray, feat_names: list[str]) -> dict[str, float]:
        meta_names = self.meta.get("feature_names_after_encode")
        if meta_names and list(meta_names) != list(feat_names):
            raise ValueError(
                f"特征名与 nas_hw_mlp_meta.json 不一致: 当前 {feat_names!r} vs meta {list(meta_names)!r}"
            )
        X_s = self.scaler_x.transform(X).astype(np.float32)
        with torch.no_grad():
            pred_s = self.mlp_model(torch.from_numpy(X_s)).cpu().numpy()
        pred = self.scaler_y.inverse_transform(pred_s)[0]
        return {name: float(pred[i]) for i, name in enumerate(self.targets)}

    def predict_raw(self, arch_config: dict[str, Any], total_params: int) -> dict[str, float]:
        X, feat_names = self._build_features(arch_config, total_params)
        if self.backend == "xgboost":
            return self._predict_raw_xgb(X, feat_names)
        if self.backend == "mlp_per_target":
            return self._predict_raw_mlp_per_target(X, feat_names)
        if self.backend == "mlp_multi_output":
            return self._predict_raw_mlp_multi_output(X, feat_names)
        raise RuntimeError(f"未知硬件代理 backend: {self.backend}")

    def predict_normalized(
        self,
        arch_config: dict[str, Any],
        total_params: int,
        *,
        lat_min_ms: float,
        lat_max_ms: float,
        pwr_min_w: float,
        pwr_max_w: float,
        require_power: bool = True,
    ) -> dict[str, float]:
        raw = self.predict_raw(arch_config, total_params)

        lat_candidates = ("latency_p50_ms", "latency_mean_ms")
        lat_key = next((k for k in lat_candidates if k in raw), None)
        if lat_key is None:
            raise ValueError(
                f"代理未预测时延目标（当前 targets={self.targets!r}），"
                "请至少训练 latency_p50_ms 或 latency_mean_ms。"
            )
        pred_lat = float(raw[lat_key])

        pow_candidates = ("power_avg_w", "dynamic_power_w")
        pow_key = next((k for k in pow_candidates if k in raw), None)
        if pow_key is None:
            if require_power:
                raise ValueError(
                    f"代理未预测功耗目标（当前 targets={self.targets!r}）。"
                    "请至少训练 power_avg_w；若只需延迟项，可在 config_hw 设 HW_GAMMA=0。"
                )
            pred_pow = 0.0
            pow_n = 0.0
        else:
            pred_pow = float(raw[pow_key])
            pow_n = _norm(pred_pow, pwr_min_w, pwr_max_w)

        return {
            "pred_latency_ms": pred_lat,
            "pred_power_w": pred_pow,
            "latency_norm": _norm(pred_lat, lat_min_ms, lat_max_ms),
            "power_norm": pow_n,
        }
