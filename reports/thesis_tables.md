# 论文可直接引用的表格

---

## Table 1: 模型参数量与性能对比

### Markdown 版本

| 模型 | 参数量 (M) | FLOPs (G) | mAP50 (%) | mAP50-95 (%) | 训练 Epoch | 备注 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| YOLOv8s | 11.17 | ~28.6 | **53.95** | 34.01 | 50 | 教师/基准 |
| Student (NAS-YOLO) | **2.75** | ~7.2 | 43.05 | 28.90 | 50 | 无蒸馏 |
| Distill-v1 | 2.75 | ~7.2 | 47.29 | 32.01 | 50 | 仅 cls 蒸馏 |
| Distill-v2 | 2.75 | ~7.2 | 48.31 | 32.65 | 50 | cls+bbox+feat |
| **D1 (Ours)** | **2.75** | ~7.2 | **52.55** | **35.56** | **120** | **渐进式蒸馏** |

### LaTeX 版本

```latex
\begin{table}[htbp]
\centering
\caption{Comparison of model parameters and detection performance on DAIR-V2X.}
\label{tab:main_result}
\begin{tabular}{lccccc}
\toprule
Model & Params (M) & FLOPs (G) & mAP$_{50}$ (\%) & mAP$_{50:95}$ (\%) & Epochs \\
\midrule
YOLOv8s (Teacher) & 11.17 & 28.6 & 53.95 & 34.01 & 50 \\
Student (w/o distill) & 2.75 & 7.2 & 43.05 & 28.90 & 50 \\
Distill-v1 (cls only) & 2.75 & 7.2 & 47.29 & 32.01 & 50 \\
Distill-v2 (combined) & 2.75 & 7.2 & 48.31 & 32.65 & 50 \\
\textbf{D1 (Ours)} & \textbf{2.75} & \textbf{7.2} & \textbf{52.55} & \textbf{35.56} & \textbf{120} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 2: 技术指标达成情况

### Markdown 版本

| 指标 | 目标值 | 实际值 | 达成 |
|:---|:---:|:---:|:---:|
| 模型压缩比 | $\ge$ 4.0$\times$ | **4.06$\times$** | ✅ |
| mAP$_{50}$ 精度下降 | $\le$ 1.5% | **1.40%** | ✅ |
| mAP$_{50:95}$ 精度下降 | — | **-1.55%** (反而提升) | ✅ |
| 单帧推理延迟 | $<$ 30 ms | 待 TensorRT 验证 | ⏳ |

### LaTeX 版本

```latex
\begin{table}[htbp]
\centering
\caption{Technical specification fulfillment.}
\label{tab:spec}
\begin{tabular}{lccc}
\toprule
Metric & Target & Achieved & Status \\
\midrule
Model compression ratio & $\ge 4.0\times$ & $\mathbf{4.06\times}$ & \cmark \\
mAP$_{50}$ degradation & $\le 1.5\%$ & $\mathbf{1.40\%}$ & \cmark \\
mAP$_{50:95}$ degradation & — & $\mathbf{-1.55\%}$ (improved) & \cmark \\
Single-frame latency & $< 30$ ms & Pending TensorRT eval. & \tikzmark \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 3: 蒸馏方法消融实验（50 Epoch 横向对比）

### Markdown 版本

| 配置 | mAP$_{50}$ (%) | 提升 (%) | 关键差异 |
|:---|:---:|:---:|:---|
| Student baseline | 43.05 | — | 无蒸馏 |
| + cls 蒸馏 | 47.29 | +4.24 | 仅 soft target |
| + cls + bbox | 48.31 | +5.26 | bbox 监督 |
| + cls + bbox + AT feature | 48.31 | +5.26 | Attention Transfer |
| **+ 渐进式 + 100 ep** | **52.34** | **+9.29** | **三阶段调度** |
| **+ 渐进式 + 120 ep** | **52.55** | **+9.50** | **充分收敛** |

### LaTeX 版本

```latex
\begin{table}[htbp]
\centering
\caption{Ablation study of distillation strategies at 50 epochs.}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
Configuration & mAP$_{50}$ (\%) & $\Delta$ (\%) \\
\midrule
Student baseline & 43.05 & — \\
+ cls distillation & 47.29 & +4.24 \\
+ cls + bbox distillation & 48.31 & +5.26 \\
+ cls + bbox + AT feature & 48.31 & +5.26 \\
\textbf{+ Progressive (100 ep)} & \textbf{52.34} & \textbf{+9.29} \\
\textbf{+ Progressive (120 ep)} & \textbf{52.55} & \textbf{+9.50} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 4: 渐进式蒸馏三阶段权重调度

### Markdown 版本

| 阶段 | Epoch 范围 | cls $\lambda$ | bbox $\lambda$ | feature $\lambda$ | Backbone 状态 |
|:---|:---:|:---:|:---:|:---:|:---|
| Warm-up | 1–10 | 1.0 | 1.0 | 0.0 | 全量训练 |
| Ramp-up | 11–30 | 1.0 | 1.0→2.0 | 0.0→5.0 | 全量训练 |
| Convergence | 31–120 | 1.0 | 2.0 | 5.0 | 全量训练 |

### LaTeX 版本

```latex
\begin{table}[htbp]
\centering
\caption{Progressive distillation weight schedule.}
\label{tab:schedule}
\begin{tabular}{lccccc}
\toprule
Stage & Epoch Range & $\lambda_{\text{cls}}$ & $\lambda_{\text{bbox}}$ & $\lambda_{\text{feat}}$ & Backbone \\
\midrule
Warm-up & 1–10 & 1.0 & 1.0 & 0.0 & Trainable \\
Ramp-up & 11–30 & 1.0 & 1.0$\to$2.0 & 0.0$\to$5.0 & Trainable \\
Convergence & 31–120 & 1.0 & 2.0 & 5.0 & Trainable \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Table 5: D1 渐进式蒸馏每 10 Epoch 详细指标

### Markdown 版本（论文插图数据源）

| Epoch | mAP$_{50}$ (%) | mAP$_{50}$ EMA (%) | 阶段 |
|:---:|:---:|:---:|:---|
| 10 | 35.48 | 35.58 | Warm-up |
| 20 | 40.38 | 40.39 | Ramp-up |
| 30 | 44.44 | 44.46 | Ramp-up end |
| 40 | 46.89 | 46.84 | Converge |
| 50 | 47.83 | 47.85 | Converge |
| 60 | 49.49 | 49.52 | Converge |
| 70 | 50.99 | 51.09 | Converge |
| 80 | 51.99 | 52.08 | Converge |
| 90 | 52.20 | 52.13 | Converge |
| 100 | 52.34 | 52.34 | Converge |
| 110 | 52.64 | 52.64 | Converge |
| 120 | 52.54 | 52.55 | Converge |

---

## Table 6: NAS 搜索最优架构配置

### Markdown 版本

| 超参数 | 搜索空间 | 最优值 |
|:---|:---|:---|
| Kernel size | {3, 5} | 3 |
| Expand ratio | {4, 5, 6, 7} | 6 |
| Width multiplier | {0.5, 0.75, 1.0, 1.25, 1.5} | 1.25 |
| Depths (C1-C4) | d$_i$ $\in$ [2,6] | [4, 4, 2, 3] |
| CIFAR-10 test accuracy | — | 82.62% |
| Backbone params | — | 0.78 M |
| Detector total params | $\le$ 2.7916 M | 2.75 M |

### LaTeX 版本

```latex
\begin{table}[htbp]
\centering
\caption{Optimal NAS architecture (Trial \#295).}
\label{tab:nas}
\begin{tabular}{lcc}
\toprule
Hyperparameter & Search Space & Optimal \\
\midrule
Kernel size & \{3, 5\} & 3 \\
Expand ratio & \{4, 5, 6, 7\} & 6 \\
Width multiplier & \{0.5, 0.75, 1.0, 1.25, 1.5\} & 1.25 \\
Depths $[d_1, d_2, d_3, d_4]$ & $d_i \in [2, 6]$ & $[4, 4, 2, 3]$ \\
CIFAR-10 test accuracy & — & 82.62\% \\
Backbone params & — & 0.78 M \\
Detector total params & $\le$ 2.7916 M & 2.75 M \\
\bottomrule
\end{tabular}
\end{table}
```

---

*以上表格可直接复制到 LaTeX 或 Word 论文中使用。*
