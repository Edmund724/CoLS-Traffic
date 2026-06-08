# CoLS-Traffic: Dual-System Agentic Collaboration via Large–Small Models for Edge–Cloud Perception in Intelligent Transportation

> A dual-system edge–cloud collaborative perception framework for Vehicle-to-Everything (V2X) scenarios
>
> **CoLS**: Collaboration via Large–Small Models

**English** | [简体中文](README_zh.md)

---

## Overview

In Intelligent Transportation Systems (ITS), high-accuracy deep neural networks incur prohibitive latency and computational costs for edge deployment, whereas lightweight models exhibit limited generalization under complex or rare traffic scenarios. This accuracy–latency–efficiency trilemma constitutes a persistent bottleneck for high-performance automated perception.

To address this challenge, we propose **CoLS-Traffic**, a dual-system collaborative perception framework that synergistically leverages a cloud-based large model (generalist) and an FPGA-class edge small model (specialist). A pretrained vision–language model (VLM) evaluates scene complexity and routes simple cases to the ultra-fast edge detector while escalating complex or uncertain instances to the cloud model. A confidence-based fallback mechanism further ensures reliability by automatically forwarding low-confidence edge predictions for cloud review. The complete pipeline encompasses hardware-aware Neural Architecture Search (NAS), progressive knowledge distillation, mixed-precision quantization-aware training (QAT), algorithm–chip co-optimized TensorRT deployment, and agentic cloud–edge collaboration.

**Core Technical Pipeline:**

```
DAIR-V2X-V Vehicle-Side Dataset
      ↓
Hardware-Aware NAS (Bayesian Optimization)    →  Optimal Backbone (0.78M params)
      ↓
Progressive Multi-Task Knowledge Distillation →  NAS-YOLO Detector (2.75M, mAP@50 52.55%)
      ↓
Mixed-Precision QAT (INT8 + FP16)            →  TensorRT Engine (mAP@50 53.09%)
      ↓
Edge Deployment (Jetson Orin Nano Super)      →  Real-Time Inference (65.8 FPS, 7.06× speedup)
      ↓
Scene Complexity Evaluation + Dynamic Routing →  Review-Mode Cloud–Edge Collaboration (+14.0%)
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
│   ├── merged_practice_report.md   # Integrated practice report (master)
│   ├── tensorrt_deployment_and_optimization_materials.md
│   ├── dynamic_routing_strategy.md
│   ├── double_buffer_pipeline.md
│   ├── power_efficiency_calculation.md
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
| NAS Backbone Search | MBConv search space + Bayesian optimization + MLP hardware proxy | Backbone 0.78 M → detector 2.75 M; compression ratio of 4.06× vs. YOLOv8s (11.17 M) |
| Progressive Knowledge Distillation | Three-stage scheduling with classification, localization, and feature-level (AT) distillation signals | Student mAP@50: 43.05% → 52.55% (+9.50 pp) |
| Mixed-Precision QAT | Backbone/Neck INT8 + Detect Head FP16; INT8 coverage 81.3% | mAP@50 53.09% (+0.54 pp); accuracy is preserved post-quantization |
| TensorRT Deployment | INT8/FP16 mixed-precision engine + dual-CUDA-stream pipeline parallelism | 65.8 FPS, 7.06× speedup over PyTorch (9.3 FPS); per-frame latency 22.3 ms |
| Dynamic Routing | 13-dimensional scene complexity features + multi-level threshold three-tier decision | Decision latency < 5 ms; cloud fallback ratio 10%–20% |
| Cloud–Edge Review Mode | Edge TRT detector + remote Qwen3-VL-8B (reviews only low-confidence bounding boxes) | Precision@IoU = 0.5 improved by 14.0% (relative) on complex scenes; end-to-end latency < 75 ms |

---

## Datasets

- **DAIR-V2X-V**: Vehicle-side cooperative perception dataset (primary experimental dataset; 22,325 frames).
- **CIFAR-10**: Proxy dataset for the NAS search phase (60,000 images at 32 × 32 resolution).

> Dataset files are voluminous and are therefore placed in the `datasets/` directory, excluded from version control. The deployment-side dataset resides in `deploy/datasets/`.

---

## Technical Reports

The `reports/` directory contains detailed technical reports for each phase:

| Report | Content |
|--------|---------|
| `CHANGELOG.md` | Development log & version changes (authoritative) |
| `merged_practice_report.md` | Integrated practice report (NAS + distillation + QAT + deployment) |
| `tensorrt_deployment_and_optimization_materials.md` | TensorRT deployment & optimization |
| `dynamic_routing_strategy.md` | Dynamic routing strategy & threshold calibration |
| `double_buffer_pipeline.md` | Double-buffer pipeline parallelism design |
| `power_efficiency_calculation.md` | Power efficiency calculation (effective compute method) |
| `benchmark_data.json` / `figure_data.csv` | Raw benchmark data |
| `figures/` | Report images (performance curves, comparison charts, etc.) |

---

## License

This project is licensed under the [MIT License](LICENSE).
