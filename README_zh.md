# CoLS-Traffic — 面向智能交通的边缘-云协同感知系统

> 面向车路协同（V2X）场景的边缘 AI 实时感知系统
>
> **CoLS**: Collaboration via Large–Small Models

[English](README.md) | **简体中文**

---

## 项目概述

针对智能交通（ITS）场景中"高算力大模型难以边缘部署、低算力小模型泛化能力不足"的痛点，本项目构建了一套 **"云端大模型（通）+ 边缘小模型（专）"** 的协同感知系统，涵盖从神经架构搜索、模型量化到边缘部署与云边协同的全流程。

**核心技术路线：**

```
DAIR-V2X-V 车端数据集
      ↓
Hardware-aware NAS（贝叶斯优化）  →  最优骨干网络（0.78M）
      ↓
渐进式多任务知识蒸馏            →  NAS-YOLO 检测模型（2.75M，mAP50 52.55%）
      ↓
混合精度 QAT（INT8+FP16）       →  TensorRT Engine（mAP50 53.09%）
      ↓
边缘部署 (Jetson Orin Nano Super) →  实时推理（65.8 FPS，加速 7.06×）
      ↓
场景复杂度评估 + 动态路由        →  Review 模式云边协同纠错（+14.0%）
```

---

## 项目结构

```
CoLS-Traffic/
├── nas_fbnet/                      # 硬件感知 NAS 搜索（A40 开发机用）
│   ├── search_hw.py                # NAS 搜索主入口（硬件反馈）
│   ├── train_hw.py                 # 超网络训练
│   ├── hw_proxy_jetson.py          # Jetson 硬件代理（延迟/功耗预测）
│   ├── config_hw.py                # 搜索配置（硬件约束参数）
│   ├── search_space_hw.py          # 搜索空间定义
│   ├── checkpoint_naming.py        # 检查点命名工具
│   ├── dataset.py                  # CIFAR-10 数据加载
│   └── models/                     # MBConv / MobileNetV3 骨干模块
│       ├── mbconv.py
│       ├── mobilenet_v3.py
│       └── mobilenet_v3_dla.py
│
├── hw_prediction/                  # 硬件性能预测（MLP 代理模型）
│   └── models_nas_mlp_per_target/  # 延迟/功耗 MLP 预测器权重
│
├── run_search_hw.py                # NAS 搜索入口脚本
│
├── deploy/                         # Jetson 部署系统（自包含）
│   ├── trt_runtime.py              # TensorRT 双缓冲推理运行时
│   ├── infer_trt.py                # TRT 推理 + mAP 验证
│   ├── benchmark_metrics.py        # 核心指标验证
│   ├── benchmark_compare.py        # PyTorch vs TRT 速度对比
│   ├── benchmark_edge_pipeline.py  # 边缘流水线测速
│   ├── benchmark_accuracy_coop.py  # 协同纠错精度验证
│   ├── benchmark_pytorch_jetson.py # Jetson 板载 PyTorch 基线测速
│   ├── calibrate_routing_thresholds.py  # 路由阈值标定
│   ├── demo_presentation.py        # 答辩演示脚本
│   ├── debug_trt_vs_pytorch.py     # TRT vs PyTorch 差异调试
│   ├── qwen3-vl.bat                # Qwen3-VL 云端模型启动脚本
│   ├── *.engine / *.onnx / *.pt    # 模型权重与引擎文件
│   ├── nas_fbnet/                  # NAS 副本（含检测器参数估计）
│   ├── detection/                  # 检测训练/推理
│   │   ├── nas_yolo.py             # NAS-YOLO 模型定义
│   │   ├── nas_backbone_feat.py    # NAS 骨干特征提取适配
│   │   ├── backbones.py            # 骨干网络定义
│   │   ├── train_nas_yolo.py       # 训练脚本（支持知识蒸馏）
│   │   ├── run_phase1_training.py  # 分阶段训练入口
│   │   ├── export_onnx_yolo.py     # ONNX 导出
│   │   ├── inference_yolo_train_eval.py  # 训练集评估推理
│   │   └── experiment_report.py    # 实验报告生成
│   ├── edge_cloud_collab/          # 云边协同子系统
│   │   ├── edge_detector.py        # 边缘检测器（TRT 封装）
│   │   ├── complexity_evaluator.py # 场景复杂度评估
│   │   ├── dynamic_router.py       # 动态路由决策器
│   │   ├── gating_network.py       # 场景复杂度门控网络
│   │   ├── cloud_client.py         # 云端 VLM 客户端
│   │   ├── simulator.py            # 云边协同仿真器
│   │   ├── utils.py                # 工具函数
│   │   └── demo.py                 # 演示入口
│   ├── configs/edge_cloud.yaml     # 系统配置文件
│   ├── datasets/                   # 部署端数据集（软链接/副本）
│   ├── reports/                    # 部署端标定报告
│   └── results/                    # 仿真与基准测试结果 (JSON)
│
├── datasets/                       # 数据集根目录
│   ├── cifar10/                    # CIFAR-10（NAS 代理数据集）
│   ├── dair_v2x_yolo/              # DAIR-V2X YOLO 格式（训练用）
│   └── raw/dair-v2x/               # DAIR-V2X 原始数据（v2x-c / v2x-v）
│
├── results/                        # NAS 搜索试验结果
│
├── reports/                        # 技术报告与文档
│   ├── CHANGELOG.md                # 开发日志与版本变更
│   ├── graduation_thesis_report.md # 毕业设计综合报告
│   ├── merged_practice_report.md   # 综合实践报告
│   ├── qat_implementation_report.md
│   ├── tensorrt_deployment_and_optimization_materials.md
│   ├── dynamic_routing_strategy.md
│   ├── double_buffer_pipeline.md
│   ├── power_efficiency_calculation.md
│   ├── thesis_tables.md            # 论文可引用表格
│   ├── benchmark_data.json / figure_data.csv
│   └── figures/                    # 报告用图片
│
├── environment.yml                 # Conda 环境配置
└── .gitignore
```

---

## 环境配置

### 硬件要求

| 组件 | 规格 |
|------|------|
| 边缘设备 | NVIDIA Jetson Orin Nano Super（67 TOPS INT8，7–25W）|
| 开发机 GPU | NVIDIA GPU（CUDA 12.8）|
| RAM | ≥ 16 GB |

### 软件依赖

```bash
# 创建 Conda 环境
conda env create -f environment.yml
conda activate gra
```

**主要依赖：**
- Python 3.11
- PyTorch 2.11 + CUDA 12.8
- TensorRT（Jetson 板载安装）
- ONNX Runtime GPU 1.24
- Ultralytics 8.4
- scikit-learn / XGBoost（硬件代理模型）

---

## 使用指南

### 1. 神经架构搜索

```bash
# 运行硬件感知 NAS 搜索（在 A40 开发机上）
python run_search_hw.py
```

### 2. 检测模型训练

```bash
# 训练 NAS-YOLO（支持知识蒸馏，在 Jetson 上执行）
cd deploy
python detection/train_nas_yolo.py --epochs 300
```

### 3. 模型量化与导出

```bash
# 导出 ONNX（在 Jetson 上执行）
cd deploy
python detection/export_onnx_yolo.py --weights best.pt --imgsz 640

# TensorRT QAT 量化（需在 Jetson 上执行）
# 生成 qat_int8.engine（QAT 为最终部署方案，PTQ 仅用于对比实验）
```

### 4. TensorRT 推理

```bash
# 单张推理
python deploy/trt_runtime.py --engine deploy/qat_int8.engine --input test.jpg --save

# 基准测试（1000次）
python deploy/trt_runtime.py --engine deploy/qat_int8.engine --input test.jpg --benchmark 1000
```

### 5. 云边协同仿真

```bash
# 运行仿真（动态路由 vs 纯边缘 vs 纯云端）
python -m deploy.edge_cloud_collab.simulator \
  --config deploy/configs/edge_cloud.yaml
```

---

## 技术亮点

| 模块 | 方法 | 效果 |
|------|------|------|
| NAS 骨干搜索 | MBConv 搜索空间 + 贝叶斯优化 + MLP 硬件代理 | 骨干 0.78M → 检测模型 2.75M，压缩比 4.06×（vs YOLOv8s 11.17M）|
| 渐进式知识蒸馏 | 三阶段调度 + 分类/定位/特征（AT）三信号蒸馏 | 学生 mAP@50 从 43.05% → 52.55%（+9.50pp）|
| 混合精度 QAT | Backbone/Neck INT8 + Detect Head FP16，覆盖率 81.3% | mAP@50 53.09%（+0.54%），精度不降反升 |
| TensorRT 部署 | INT8/FP16 混合引擎 + 双 CUDA 流流水线并行 | 65.8 FPS，加速比 7.06×（vs PyTorch 9.3 FPS），单帧 22.3ms |
| 动态路由 | 13 维场景复杂度特征 + 多层级阈值三级判定 | 决策延迟 <5ms，云端回退比例 10%~20% |
| 云边协同 Review 模式 | 边缘 TRT + 远端 Qwen3-VL-8B（仅审查低置信度框） | 复杂场景 Precision@IoU=0.5 相对提升 14.0%，端到端 <75ms |

---

## 数据集

- **DAIR-V2X-V**：车端视角车路协同感知数据集（主实验数据集，22,325 帧）
- **CIFAR-10**：NAS 搜索阶段代理数据集（60,000 张 32×32 图像）

> 数据集文件体积较大，已放置在 `datasets/` 目录，不纳入 Git 管理。部署端使用 `deploy/datasets/` 目录。

---

## 技术报告

`reports/` 目录包含各阶段的详细技术报告：

| 报告 | 内容 |
|------|------|
| `CHANGELOG.md` | 开发日志与版本变更（权威版） |
| `graduation_thesis_report.md` | 毕业设计综合报告 |
| `merged_practice_report.md` | 综合实践报告 |
| `qat_implementation_report.md` | QAT 量化训练实现与 PTQ 对比分析 |
| `tensorrt_deployment_and_optimization_materials.md` | TensorRT 部署与优化 |
| `dynamic_routing_strategy.md` | 动态路由策略设计 |
| `double_buffer_pipeline.md` | 双缓冲流水线设计 |
| `power_efficiency_calculation.md` | 能效比计算与分析 |
| `thesis_tables.md` | 论文可直接引用的表格（Markdown + LaTeX） |
| `benchmark_data.json` / `figure_data.csv` | 基准测试原始数据 |
| `figures/` | 报告用图片（性能曲线、对比图表等） |

---

## License

This project is licensed under the [MIT License](LICENSE).
