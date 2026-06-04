#!/usr/bin/env python3
"""硬件反馈 NAS 入口（独立脚本，不修改 run_search.py / search.py）。

使用前：
  1) 在 hw_prediction 下训练好 XGBoost 代理（preprocess.joblib + xgb_*.json）
  2) 按需编辑 nas_fbnet/config_hw.py（权重 α/β/γ、归一化范围、输出目录）

运行（在项目根目录 pytorch_nas_jetson）::
  python run_search_hw.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_fbnet.search_hw import run_search_hw

if __name__ == "__main__":
    run_search_hw()
