"""
硬件反馈 NAS 专用配置：与 config.py 独立，修改本文件即可调 HW 权重与路径。

参考 tf_nas_fpga/nas_train_qkeras_HW.py：
  combined = α·(1 - val_acc) + β·latency_norm + γ·power_norm
贝叶斯优化对该组合目标做最小化（与原版「最小化 -val_acc」不同，请勿混用同一 search_cache）。
"""
import os

# 当前文件所在目录：`nas_fbnet/`
_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：`pytorch_nas_jetson/`
PROJECT_ROOT = os.path.dirname(_DIR)

# CIFAR-10 数据集根目录；`dataset.py`/`dataset_hw.py` 会从这里读取数据
CIFAR10_ROOT = os.path.join(PROJECT_ROOT, "datasets", "cifar10", "cifar-10-batches-py")

# 搜索阶段总调用次数；贝叶斯优化会评估这么多个候选架构
SEARCH_CALLS = 2000
# 每个候选架构在搜索阶段训练的最大 epoch 数
SEARCH_EPOCHS = 100
# 搜索阶段早停 patience；验证集长期不提升就提前结束
SEARCH_PATIENCE = 15

# 搜索完成后，对最佳架构做完整重训练的最大 epoch 数
RETRAIN_EPOCHS = 400
# 重训练阶段早停 patience
RETRAIN_PATIENCE = 15

# 默认训练 batch size；用于完整训练/重训练
BATCH_SIZE = 2048
# 搜索阶段训练时的 batch size；通常与 BATCH_SIZE 保持一致
SEARCH_BATCH_SIZE = 2048
# SGD 初始学习率
INITIAL_LR = 0.1
# L2 正则项系数
WEIGHT_DECAY = 5e-4
# SGD 动量
MOMENTUM = 0.9
# 标签平滑系数
LABEL_SMOOTHING = 0.1
# warmup epoch 数；前几轮逐步把学习率拉到初始值
WARMUP_EPOCHS = 5
# Random Erasing 数据增强概率
RANDOM_ERASING_P = 0.25
# 搜索阶段是否保存每个 trial 的训练/验证曲线图
SEARCH_SAVE_PLOTS = True

# 各 stage 的输出通道配置；需与 NUM_STAGES 和网络实现匹配
STAGE_WIDTHS = [16, 72, 56, 40, 32]
# stem 输出通道数
INPUT_CHANNEL = 16
# head 前最后一层通道数
LAST_CHANNEL = 256
# 分类类别数；CIFAR-10 固定为 10
NUM_CLASSES = 10
# 分类 head dropout 概率
DROPOUT_RATE = 0.2
# 是否启用面向 DLA 部署的 head 结构
USE_DLA_HEAD = True
# 输入图像边长；当前搜索默认针对 CIFAR-10 的 32x32
CIFAR_IMAGE_SIZE = 32

# 可选卷积核尺寸列表
KS_LIST = [3, 5, 7]
# 可选 expansion ratio 列表
EXPAND_RATIO_LIST = [3, 4, 5, 6, 7]
# 可选宽度倍率列表
WIDTH_MULTIPLIER_LIST = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
# 每个 stage 的最小深度
DEPTH_MIN = 2
# 每个 stage 的最大深度
DEPTH_MAX = 4
# 可搜索 stage 数；需与特征编码/硬件代理保持一致
NUM_STAGES = 4

# 推理部署模式搜索空间；需与 engine 构建和硬件代理训练数据一致
INFER_MODE_LIST = ["only-gpu", "only-dla", "dla1gpu", "dla2gpu", "dla3gpu"]
# 量化精度搜索空间
QUANT_PRECISION_LIST = ["fp16", "int8"]

# 组合目标权重：
#   total_loss = α * (1 - val_acc) + β * latency_norm + γ * power_norm
# 三者越大，对应项在搜索目标中的影响越强
HW_ALPHA = 0.5
HW_BETA = 0.25
HW_GAMMA = 0.25

# 硬件反馈搜索结果输出目录；与原版搜索目录分开，避免互相覆盖
OUTPUT_DIR = os.path.join(PROJECT_ROOT, f"nas_fbnet_outputs_hw-{HW_ALPHA}_{HW_BETA}_{HW_GAMMA}")
# 当前最优候选架构的导出路径
BEST_CONFIG_PATH = os.path.join(OUTPUT_DIR, "best_config_hw.json")
# 贝叶斯优化缓存；不要与纯精度搜索共用
SEARCH_CACHE_PATH = os.path.join(OUTPUT_DIR, "search_cache_hw.json")
# 每次搜索评估的日志 CSV
SEARCH_CSV_PATH = os.path.join(OUTPUT_DIR, "search_log_hw.csv")

# 硬件代理模型目录：
#   - 优先支持 nas_hw_mlp_per_target.py 产物（mlp_<target>.pt 等）
#   - 兼容旧的 nas_hw_mlp.py / nas_hw_xgboost.py 产物
# 修改这里即可切换当前 NAS 搜索使用的硬件代理模型版本
HW_PROXY_MODEL_DIR = os.path.join(PROJECT_ROOT, "hw_prediction", "models_nas_mlp_per_target")


# 代理预测值线性归一化到 [0,1] 的上下界。
# 建议按当前 `trt_benchmark_results.csv` 的真实分布维护，
# 否则 latency/power 在组合目标中的相对权重会失真。
HW_LATENCY_MIN_MS = 0.5
HW_LATENCY_MAX_MS = 3.0
HW_POWER_MIN_W = 8.0
HW_POWER_MAX_W = 11.0

# 如果硬件代理预测平均延迟超过该阈值，则直接跳过该候选训练并返回惩罚损失。
# 0 表示关闭该剪枝规则。
HW_MAX_PRED_LATENCY_MS = 0.0

# 当候选被跳过、训练失败或硬件代理异常时，返回给 skopt 的大惩罚值。
# 数值越大，优化器越不可能继续靠近该区域。
HW_SKIP_LOSS = 5000.0
