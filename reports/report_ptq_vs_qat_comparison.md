# PTQ vs QAT 量化精度对比实验报告

> 实验日期：2026-04-23  
> 实验目的：验证在 NAS-YOLOv8 学生模型上，直接 PTQ（仅校准、不微调）与 QAT（校准 + 量化感知微调）对目标检测精度的影响，为后续 TensorRT INT8 部署提供量化策略依据。

---

## 1. 实验背景

在完成知识蒸馏（第2阶段）后，学生模型已具备接近教师的精度（mAP50≈0.5255）。第3阶段的目标是在不显著损失精度的前提下，将模型压缩至 INT8 以适配 Jetson Orin Nano 的 INT8 Tensor Core。

量化有两种主流路径：
- **PTQ（Post-Training Quantization）**：在训练好的 FP32 模型上插入量化节点，仅通过校准集统计激活值的动态范围（amax），无需反向传播。
- **QAT（Quantization-Aware Training）**：在 PTQ 校准的基础上，继续以量化后的前向/反向传播进行微调，使权重适应低精度表示的噪声。

本实验通过控制变量（同一初始化权重、同一校准集、同一评估协议），严格对比 PTQ 与 QAT 的精度差异。

---

## 2. 实验环境与配置

| 项目 | 配置 |
|:---|:---|
| GPU | NVIDIA A40 (CUDA 12.4) |
| PyTorch | 2.6.0+cu124 |
| 量化库 | `pytorch-quantization==2.1.3` (NVIDIA) |
| 评估库 | `torchmetrics[detection]` |
| 数据集 | DAIR-V2X (nc=10, train=11163, val=4464) |
| 输入尺寸 | 640×640 |
| 评估阈值 | conf=0.25, IoU=0.45 |

### 2.1 模型信息

- **架构**：NAS-YOLOv8（硬件感知 NAS 搜索所得）
- **参数量**：~2.75 M（约为 YOLOv8s 的 1/4，满足 >4× 压缩比指标）
- **基准权重**：`distillation/runs/distill_dair/distill_d1_at/best_ema.pt`

### 2.2 混合精度量化策略

为平衡精度与加速比，采用**分层混合精度**策略：

| 模块 | 精度 | 说明 |
|:---|:---|:---|
| Backbone（特征提取） | INT8 | 占参数量 ~62%，对精度影响较小 |
| Neck（FPN/PAN 融合） | INT8 | 占参数量 ~19%，计算密集 |
| **Detect Head**（检测头） | **FP16/FP32** | cv2（bbox 分支）、cv3（cls 分支）、DFL 均不量化 |

- **量化覆盖率**：约 **81.3%** 的参数量被量化至 INT8。
- **Detect Head 保留 FP16 的原因**：bbox 回归与 DFL 分布 focal loss 对数值精度极为敏感，直接 INT8 量化会导致坐标偏移和类别置信度坍塌。

### 2.3 QAT 训练超参

| 超参 | 设置 |
|:---|:---|
| 优化器 | AdamW |
| 学习率 | 1e-4（CosineAnnealingLR, eta_min=1e-6） |
| 权重衰减 | 5e-4 |
| Batch Size | 16 |
| Epochs | 30 |
| AMP | **关闭**（QAT 与 fp16 autocast 不兼容） |
| 校准批次 | 256 batches（来自训练集） |

---

## 3. 实验脚本说明

### 3.1 核心脚本

| 脚本 | 功能 |
|:---|:---|
| `qat_train.py` | QAT 完整流水线：加载蒸馏模型 → 插入 QuantConv2d → 禁用 Head 量化 → 校准 → 30 epoch QAT 微调 → 保存 checkpoint → 导出 QDQ-ONNX |
| `qat_ptq_compare.py` | **对照实验脚本**：在同一进程中依次评估 (1) FP32 基线 → (2) PTQ（仅校准，不训练）→ (3) QAT（校准+微调），确保数据加载器、评估指标、NMS 参数完全一致 |

### 3.2 `qat_ptq_compare.py` 实验流程

```
FP32 Baseline
  └─ 加载 best_ema.pt，BN 设为 eval，直接跑 val mAP

PTQ Branch
  └─ 加载同一 best_ema.pt
  └─ quant_modules.initialize() 全局替换 Conv2d → QuantConv2d
  └─ 禁用 Detect Head 中 25 个 quantizer
  └─ 冻结 BN running stats
  └─ 256 batches MaxCalibrator 前向校准（load_calib_amax, strict=False）
  └─ 直接评估 val mAP（无反向传播）

QAT Branch
  └─ 加载 qat_runs/qat_dair_v1/best.pt（30 epoch 微调后的 checkpoint）
  └─ 重新禁用 Detect Head quantizer（加载时会重新初始化并启用）
  └─ 冻结 BN running stats
  └─ 评估 val mAP
```

### 3.3 评估指标计算方式

- 使用 `torchmetrics.detection.MeanAveragePrecision`（IoU_type="bbox", box_format="xyxy"）。
- 后处理：对模型输出的 `(bs, 4+nc, num_anchors)` 张量逐图解析：
  1. xywh → xyxy 转换并 clamp 到图像边界；
  2. 逐类别取最大置信度，按 conf>0.25 过滤；
  3. 使用 `torchvision.ops.nms`（IoU=0.45, class-aware offset）。
- 空预测处理：若某图无检测框，向 metric 更新空的 `preds=[]`，避免 torchmetrics 报错。

---

## 4. 实验结果

### 4.1 精度对比

| 模型 | mAP50 | mAP50-95 | 相对 FP32 (mAP50) | 绝对 drop (mAP50) |
|:---|:---|:---|:---|:---|
| **FP32 基线** | **0.5255** | **0.3557** | — | — |
| **PTQ（仅校准）** | **0.5249** | **0.3546** | **-0.05%** | -0.0006 |
| **QAT（校准+30 epoch 微调）** | **0.5309** | **0.3595** | **+0.54%** | +0.0054 |

### 4.2 结果可视化

```
mAP50
 0.531 ┤                              ┌─── QAT (0.5309)
 0.529 ┤
 0.527 ┤
 0.525 ┤──────── FP32 (0.5255)
       │
 0.524 ┤         PTQ (0.5249)
       └────────────────────────────────────
              FP32        PTQ         QAT
```

### 4.3 关键发现

1. **PTQ 精度损失极小**
   - mAP50 仅下降 0.0006（相对 -0.05%），mAP50-95 下降 0.0011（相对 -0.31%）。
   - 这表明当前模型架构（NAS 搜索所得的深度/宽度比例）与 DAIR-V2X 数据分布对 INT8 量化具有**良好的天然鲁棒性**。
   - 混合精度策略（Head 保留 FP16）起到了决定性作用：若将 Detect Head 也量化，预期会出现显著掉点。

2. **QAT 的提升主要来自"继续微调"而非量化本身**
   - QAT 阶段未使用教师蒸馏损失，仅以标准 v8DetectionLoss（box + cls + dfl）进行 30 epoch 低学习率微调。
   - 蒸馏阶段在 epoch 120 时已收敛至 mAP50≈0.523–0.526 的瓶颈；QAT 的继续优化使模型跳出了局部最优，达到 0.5309。
   - 因此，**QAT 在此场景下更像是一次"无约束的继续训练"**，而非"量化噪声的补偿训练"。

3. **校准过程的稳定性**
   - MaxCalibrator 在 256 batches（约 4k 张图）上统计激活 amax，覆盖了数据集中主要场景（白天/夜晚/城市道路/高速/遮挡）。
   - Detect Head 中被禁用的 quantizer 会输出 `Calibrator returned None... Set amax to NaN` 警告，这是预期行为（这些层不参与量化），不影响结果。

---

## 5. 分析与讨论

### 5.1 为什么 PTQ 已经足够好？

- **NAS 搜索的架构优势**：硬件感知 NAS 在搜索阶段已将"低比特友好性"隐含地纳入了优化目标（通过限制通道数为 8/16 的整数倍、避免极端深度的残差连接），使得最终架构的激活分布自然平滑，极值点少。
- **数据分布稳定**：DAIR-V2X 为车路协同数据集，相机视角固定、光照变化范围相对可控，激活统计量在校准集上具有代表性。
- **混合精度的安全边际**：将最敏感的 Head 保留在 FP16，避免了量化误差在 bbox 回归上的累积放大。

### 5.2 QAT 是否必要？

| 场景 | 建议 |
|:---|:---|
| **追求极致部署速度**，可接受 <0.1% 精度损失 | **PTQ 足够**，无需额外训练开销 |
| **论文/指标要求"精度零损失或提升"** | **QAT 更稳妥**，30 epoch 微调后精度反而略有提升 |
| **后续出现域迁移**（如从 DAIR-V2X 迁移到 BDD100K）| **必须 QAT**，不同数据分布下 PTQ 掉点会显著增大 |

### 5.3 与任务书技术指标的对照

| 指标 | 要求 | 当前状态 |
|:---|:---|:---|
| mAP 下降幅度 | < 1.5% | PTQ: **0.05%** ✓; QAT: **-0.54%（提升）** ✓ |
| 参数量压缩比 | > 4× | **~4×** (vs YOLOv8s) ✓ |
| 边缘端延迟 | < 30 ms | 待 TensorRT 部署后实测 |
| 能效比 | ≥ 2 TOPS/W | 待 Jetson 实测 |

---

## 6. 产物与路径

| 文件 | 路径 | 说明 |
|:---|:---|:---|
| QAT 训练脚本 | `qat_train.py` | 完整 QAT 流水线 + QDQ-ONNX 导出 |
| 对照实验脚本 | `qat_ptq_compare.py` | 三模型严格对照评估 |
| QAT 最佳权重 | `qat_runs/qat_dair_v1/best.pt` | 含 180 个 `_amax` 量化参数 |
| QAT 最终权重 | `qat_runs/qat_dair_v1/last.pt` | 第 30 epoch |
| QDQ-ONNX | `qat_runs/qat_dair_v1/model_qdq.onnx` | 256 个 QDQ 节点，TensorRT 兼容 |
| 实验日志 | `qat_runs/qat_dair_v1/ptq_vs_qat.log` | 三模型 mAP 汇总 |
| 训练日志 | `qat_runs/qat_dair_v1/` | 各 epoch 的 loss / mAP / lr |

---

## 7. 后续建议

1. **TensorRT INT8 编译与本地测速**
   - 使用 `trtexec --int8 --fp16 --onnx=model_qdq.onnx` 生成 engine。
   - 在 A40 上先验证 ONNX→TRT 的推理延迟，作为 Jetson 部署前的 sanity check。

2. **Jetson Orin Nano 实测**
   - 将 `model_qdq.onnx` 拷贝至 Jetson，使用 TensorRT 8.6+ 编译。
   - 测量 Batch=1 的端到端延迟、FPS、功耗（`jtop` / `tegrastats`）。
   - 验证是否满足 <30 ms /帧 的实时性指标。

3. **PTQ vs QAT Engine 对比**
   - 分别导出 PTQ-only（不加载 QAT 微调权重）和 QAT 的 ONNX，编译为 TRT engine。
   - 在 Jetson 上对比两者的延迟差异（理论上几乎无差异，因为网络结构相同，仅权重/scale 不同）。

4. **云端协同仿真**
   - 搭建 Qwen3-VL-8B 场景复杂度评估器 + 动态路由策略。
   - 在复杂场景（雨夜、遮挡）下验证"云边协同"相比纯边缘推理的准确率提升（目标 >10%）。

---

## 附录 A：`qat_ptq_compare.py` 关键代码片段

```python
# 1. FP32 基线
model_fp32 = NASYOLOv8.load(orig_weights).to(device).eval()

# 2. PTQ：校准后不做任何训练
quant_modules.initialize()
model_ptq = build_ptq_model(orig_weights, device, train_loader, calib_batches=256)
# build_ptq_model 内部：插入 quantizer → 禁用 Head → 冻结 BN → 256 batch 校准

# 3. QAT：加载微调后的权重
model_qat = NASYOLOv8.load(qat_weights).to(device).eval()
# 关键：加载后必须重新禁用 Detect Head 的 quantizer
for name, mod in model_qat.detect.named_modules():
    if hasattr(mod, "_input_quantizer"):
        mod._input_quantizer.disable()
        mod._weight_quantizer.disable()
```

---

*报告生成完毕。如有需要补充 TensorRT 测速结果或 Jetson 实测数据，可在本报告后续章节追加。*
