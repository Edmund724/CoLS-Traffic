# GRA — 面向智能交通的边缘-云协同感知系统

> **毕业设计项目** · 面向车路协同（V2X）场景的边缘 AI 实时感知系统

---

## 项目概述

针对智能交通（ITS）场景中"高算力大模型难以边缘部署、低算力小模型泛化能力不足"的痛点，本项目构建了一套 **"云端大模型（通）+ 边缘小模型（专）"** 的协同感知系统，涵盖从神经架构搜索、模型量化到边缘部署与云边协同的全流程。

**核心技术路线：**

```
DAIR-V2X 数据集
      ↓
Hardware-aware NAS (FBNet)  →  搜索最优骨干网络
      ↓
YOLO 检测头训练（知识蒸馏）  →  NAS-YOLO 检测模型
      ↓
PTQ / QAT 量化（INT8）       →  TensorRT Engine
      ↓
边缘部署 (Jetson Orin Nano)  →  实时推理
      ↓
场景复杂度门控 + 动态路由    →  云边协同纠错
```

---

## 项目结构

```
gra/
├── nas_fbnet/              # 硬件感知神经架构搜索（FBNet）
│   ├── search_hw.py        # NAS 搜索主入口
│   ├── train_hw.py         # 超网络训练
│   ├── hw_proxy_jetson.py  # Jetson 硬件代理（延迟/功耗测量）
│   ├── search_space_hw.py  # 搜索空间定义
│   └── models/             # MBConv / MobileNetV3 等模块
│
├── nas_jetson_backbone/    # Jetson 骨干网络搜索（早期版本）
│   ├── run_search.py
│   └── train_dla.py
│
├── detection_bdd100k/      # BDD100K 检测模型训练
│   ├── nas_yolo.py         # NAS-YOLO 模型定义
│   ├── train_nas_yolo.py   # 训练脚本（支持知识蒸馏）
│   └── cache_teacher_outputs.py
│
├── deploy/                 # 部署模块
│   ├── trt_runtime.py      # TensorRT 双缓冲推理运行时
│   ├── infer_trt.py        # TensorRT 推理脚本
│   ├── benchmark_metrics.py # 精度/速度基准评测
│   ├── edge_cloud_collab/  # 云边协同子系统
│   │   ├── edge_detector.py       # 边缘检测器（TRT封装）
│   │   ├── complexity_evaluator.py # 场景复杂度评估
│   │   ├── dynamic_router.py      # 动态路由决策器
│   │   ├── cloud_client.py        # 云端 VLM 客户端
│   │   ├── simulator.py           # 云边协同仿真器
│   │   └── demo.py                # 演示入口
│   └── configs/edge_cloud.yaml    # 系统配置文件
│
├── hw_prediction/          # 硬件性能预测（MLP代理模型）
│   └── models_nas_mlp_per_target/
│
├── reports/                # 技术报告与文档
│   ├── CHANGELOG.md
│   └── figures/
│
└── environment.yml         # Conda 环境配置
```

---

## 环境配置

### 硬件要求

| 组件 | 规格 |
|------|------|
| 边缘设备 | NVIDIA Jetson Orin Nano Super（40 TOPS）|
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
# 运行硬件感知 NAS 搜索
python nas_fbnet/search_hw.py \
  --config nas_fbnet/config_hw.py \
  --dataset datasets/cifar10
```

### 2. 检测模型训练

```bash
# 训练 NAS-YOLO（支持知识蒸馏）
python detection_bdd100k/train_nas_yolo.py \
  --data detection_bdd100k/bdd100k.yaml \
  --epochs 300
```

### 3. 模型量化与导出

```bash
# 导出 ONNX
python detection_bdd100k/export_onnx_yolo.py --weights best.pt --imgsz 640

# TensorRT PTQ / QAT 量化（需在 Jetson 上执行）
# 生成 ptq_int8.engine / qat_int8.engine
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
| NAS 骨干搜索 | FBNet + 硬件代理（延迟/功耗约束） | 自动生成适配 Jetson 的最优子网络 |
| 知识蒸馏 | Teacher-Student 框架（YOLOv8 → NAS-YOLO） | mAP 损失 < 1.5% |
| 量化感知训练 | QAT INT8（vs PTQ） | 精度无损，推理速度 ↑ |
| TensorRT 部署 | 双缓冲流水线 + CUDA Graph | 吞吐提升 5×+ |
| 动态路由 | 场景复杂度门控（规则/MLP） | 复杂场景识别率 ↑ 10%+ |
| 云边协同 | 本地 TRT + 远端 VLM（Qwen3-VL） | 端到端 < 100ms（简单场景） |

---

## 数据集

- **DAIR-V2X**：车路协同感知数据集（主实验数据集）
- **BDD100K**：自动驾驶感知数据集（辅助训练）
- **CIFAR-10**：NAS 搜索阶段代理数据集

> 数据集文件体积较大，不纳入 Git 管理，请按 `datasets/` 目录结构自行下载放置。

---

## 技术报告

`reports/` 目录包含各阶段的详细技术报告：

| 报告 | 内容 |
|------|------|
| `CHANGELOG.md` | 开发日志与版本变更 |
| `qat_implementation_report.md` | QAT 量化训练实现细节 |
| `tensorrt_deployment_and_optimization_materials.md` | TensorRT 部署与优化 |
| `dynamic_routing_strategy.md` | 动态路由策略设计 |
| `double_buffer_pipeline.md` | 双缓冲流水线设计 |
| `power_efficiency_calculation.md` | 能效比计算与分析 |
| `graduation_thesis_report.md` | 毕业设计综合报告 |

---

## License

Private — 仅供学术研究与毕业设计使用。
