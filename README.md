# CoLS-Traffic — Edge-Cloud Collaborative Perception for Intelligent Traffic

> An edge AI real-time perception system for Vehicle-to-Everything (V2X) scenarios
>
> **CoLS**: Collaboration via Large–Small Models

**English** | [简体中文](README_zh.md)

---

## Overview

To address the dilemma in Intelligent Transportation Systems (ITS) where *high-compute large models are difficult to deploy on edge devices while low-compute small models lack generalization*, this project builds a **"Cloud Large Model (Generalist) + Edge Small Model (Specialist)"** collaborative perception system, covering the full pipeline from Neural Architecture Search and model quantization to edge deployment and cloud-edge collaboration.

**Core Technical Pipeline:**

```
DAIR-V2X-V Vehicle-Side Dataset
      ↓
Hardware-aware NAS (Bayesian Optimization)  →  Optimal Backbone (0.78M params)
      ↓
Progressive Multi-task Knowledge Distillation →  NAS-YOLO Detector (2.75M, mAP50 52.55%)
      ↓
Mixed-precision QAT (INT8+FP16)            →  TensorRT Engine (mAP50 53.09%)
      ↓
Edge Deployment (Jetson Orin Nano Super)   →  Real-time Inference (65.8 FPS, 7.06× speedup)
      ↓
Scene Complexity Evaluation + Dynamic Routing →  Review-mode Cloud-Edge Collaboration (+14.0%)
```

---

## Project Structure

```
CoLS-Traffic/
├── nas_fbnet/                      # Hardware-aware NAS search (A40 dev server)
│   ├── search_hw.py                # NAS search entry (hardware feedback)
│   ├── train_hw.py                 # Supernet training
│   ├── hw_proxy_jetson.py          # Jetson hardware proxy (latency/power prediction)
│   ├── config_hw.py                # Search config (hardware constraint params)
│   ├── search_space_hw.py          # Search space definition
│   ├── checkpoint_naming.py        # Checkpoint naming utility
│   ├── dataset.py                  # CIFAR-10 data loading
│   └── models/                     # MBConv / MobileNetV3 backbone modules
│       ├── mbconv.py
│       ├── mobilenet_v3.py
│       └── mobilenet_v3_dla.py
│
├── hw_prediction/                  # Hardware performance prediction (MLP proxy model)
│   └── models_nas_mlp_per_target/  # Latency/power MLP predictor weights
│
├── run_search_hw.py                # NAS search entry script
│
├── deploy/                         # Jetson deployment system (self-contained)
│   ├── trt_runtime.py              # TensorRT double-buffer inference runtime
│   ├── infer_trt.py                # TRT inference + mAP validation
│   ├── benchmark_metrics.py        # Core metric validation
│   ├── benchmark_compare.py        # PyTorch vs TRT speed comparison
│   ├── benchmark_edge_pipeline.py  # Edge pipeline benchmarking
│   ├── benchmark_accuracy_coop.py  # Collaborative correction accuracy validation
│   ├── benchmark_pytorch_jetson.py # On-device PyTorch baseline benchmarking
│   ├── calibrate_routing_thresholds.py  # Routing threshold calibration
│   ├── demo_presentation.py        # Demo presentation script
│   ├── debug_trt_vs_pytorch.py     # TRT vs PyTorch diff debugging
│   ├── qwen3-vl.bat                # Qwen3-VL cloud model launch script
│   ├── *.engine / *.onnx / *.pt    # Model weights & engine files
│   ├── nas_fbnet/                  # NAS copy (with detector param estimation)
│   ├── detection/                  # Detection training/inference
│   │   ├── nas_yolo.py             # NAS-YOLO model definition
│   │   ├── nas_backbone_feat.py    # NAS backbone feature extraction adapter
│   │   ├── backbones.py            # Backbone network definitions
│   │   ├── train_nas_yolo.py       # Training script (supports knowledge distillation)
│   │   ├── run_phase1_training.py  # Phased training entry
│   │   ├── export_onnx_yolo.py     # ONNX export
│   │   ├── inference_yolo_train_eval.py  # Training set evaluation inference
│   │   └── experiment_report.py    # Experiment report generation
│   ├── edge_cloud_collab/          # Cloud-edge collaboration subsystem
│   │   ├── edge_detector.py        # Edge detector (TRT wrapper)
│   │   ├── complexity_evaluator.py # Scene complexity evaluation
│   │   ├── dynamic_router.py       # Dynamic routing decision maker
│   │   ├── gating_network.py       # Scene complexity gating network
│   │   ├── cloud_client.py         # Cloud VLM client
│   │   ├── simulator.py            # Cloud-edge collaboration simulator
│   │   ├── utils.py                # Utility functions
│   │   └── demo.py                 # Demo entry
│   ├── configs/edge_cloud.yaml     # System config file
│   ├── datasets/                   # Deployment-side dataset (symlink/copy)
│   ├── reports/                    # Deployment-side calibration reports
│   └── results/                    # Simulation & benchmark results (JSON)
│
├── datasets/                       # Dataset root directory
│   ├── cifar10/                    # CIFAR-10 (NAS proxy dataset)
│   ├── dair_v2x_yolo/              # DAIR-V2X YOLO format (for training)
│   └── raw/dair-v2x/               # DAIR-V2X raw data (v2x-c / v2x-v)
│
├── results/                        # NAS search experiment results
│
├── reports/                        # Technical reports & documentation
│   ├── CHANGELOG.md                # Development log & version changes
│   ├── graduation_thesis_report.md # Graduation thesis comprehensive report
│   ├── merged_practice_report.md   # Integrated practice report
│   ├── qat_implementation_report.md
│   ├── tensorrt_deployment_and_optimization_materials.md
│   ├── dynamic_routing_strategy.md
│   ├── double_buffer_pipeline.md
│   ├── power_efficiency_calculation.md
│   ├── thesis_tables.md            # Paper-citable tables
│   ├── benchmark_data.json / figure_data.csv
│   └── figures/                    # Report images
│
├── environment.yml                 # Conda environment config
└── .gitignore
```

---

## Environment Setup

### Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Edge Device | NVIDIA Jetson Orin Nano Super (67 TOPS INT8, 7–25W) |
| Dev Server GPU | NVIDIA GPU (CUDA 12.8) |
| RAM | ≥ 16 GB |

### Software Dependencies

```bash
# Create Conda environment
conda env create -f environment.yml
conda activate gra
```

**Key Dependencies:**
- Python 3.11
- PyTorch 2.11 + CUDA 12.8
- TensorRT (installed on Jetson)
- ONNX Runtime GPU 1.24
- Ultralytics 8.4
- scikit-learn / XGBoost (hardware proxy model)

---

## Usage Guide

### 1. Neural Architecture Search

```bash
# Run hardware-aware NAS search (on A40 dev server)
python run_search_hw.py
```

### 2. Detection Model Training

```bash
# Train NAS-YOLO (with knowledge distillation, on Jetson)
cd deploy
python detection/train_nas_yolo.py --epochs 300
```

### 3. Model Quantization & Export

```bash
# Export ONNX (on Jetson)
cd deploy
python detection/export_onnx_yolo.py --weights best.pt --imgsz 640

# TensorRT QAT quantization (must be run on Jetson)
# Generates qat_int8.engine (QAT is the final deployment scheme; PTQ is for comparison only)
```

### 4. TensorRT Inference

```bash
# Single image inference
python deploy/trt_runtime.py --engine deploy/qat_int8.engine --input test.jpg --save

# Benchmark (1000 iterations)
python deploy/trt_runtime.py --engine deploy/qat_int8.engine --input test.jpg --benchmark 1000
```

### 5. Cloud-Edge Collaboration Simulation

```bash
# Run simulation (dynamic routing vs. edge-only vs. cloud-only)
python -m deploy.edge_cloud_collab.simulator \
  --config deploy/configs/edge_cloud.yaml
```

---

## Technical Highlights

| Module | Method | Result |
|--------|--------|--------|
| NAS Backbone Search | MBConv search space + Bayesian optimization + MLP hardware proxy | Backbone 0.78M → detector 2.75M, compression ratio 4.06× (vs YOLOv8s 11.17M) |
| Progressive Knowledge Distillation | 3-stage scheduling + classification/localization/feature (AT) triple-signal distillation | Student mAP@50 from 43.05% → 52.55% (+9.50pp) |
| Mixed-precision QAT | Backbone/Neck INT8 + Detect Head FP16, coverage 81.3% | mAP@50 53.09% (+0.54%), accuracy improves rather than degrades |
| TensorRT Deployment | INT8/FP16 mixed engine + dual CUDA stream pipeline parallelism | 65.8 FPS, speedup 7.06× (vs PyTorch 9.3 FPS), 22.3ms per frame |
| Dynamic Routing | 13-dim scene complexity features + multi-level threshold 3-tier decision | Decision latency <5ms, cloud fallback ratio 10%–20% |
| Cloud-Edge Review Mode | Edge TRT + remote Qwen3-VL-8B (reviews only low-confidence boxes) | Complex scene Precision@IoU=0.5 improved by 14.0% relatively, E2E <75ms |

---

## Datasets

- **DAIR-V2X-V**: Vehicle-side V2X perception dataset (primary experimental dataset, 22,325 frames)
- **CIFAR-10**: NAS search proxy dataset (60,000 32×32 images)

> Dataset files are large and placed in the `datasets/` directory, excluded from Git tracking. The deployment-side uses `deploy/datasets/`.

---

## Technical Reports

The `reports/` directory contains detailed technical reports for each phase:

| Report | Content |
|--------|---------|
| `CHANGELOG.md` | Development log & version changes (authoritative) |
| `graduation_thesis_report.md` | Graduation thesis comprehensive report |
| `merged_practice_report.md` | Integrated practice report |
| `qat_implementation_report.md` | QAT training implementation & PTQ comparison |
| `tensorrt_deployment_and_optimization_materials.md` | TensorRT deployment & optimization |
| `dynamic_routing_strategy.md` | Dynamic routing strategy design |
| `double_buffer_pipeline.md` | Double-buffer pipeline design |
| `power_efficiency_calculation.md` | Power efficiency calculation & analysis |
| `thesis_tables.md` | Paper-citable tables (Markdown + LaTeX) |
| `benchmark_data.json` / `figure_data.csv` | Raw benchmark data |
| `figures/` | Report images (performance curves, comparison charts, etc.) |

---

## License

This project is licensed under the [MIT License](LICENSE).
