"""NAS-FBNet 配置"""
import os

# 数据路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CIFAR10_ROOT = os.path.join(PROJECT_ROOT, "datasets", "cifar10", "cifar-10-batches-py")

# 搜索阶段
SEARCH_CALLS = 2000         # 贝叶斯优化评估次数
SEARCH_EPOCHS = 200   # 每个候选架构的代理训练轮数
SEARCH_PATIENCE = 15        # 早停：val_acc 连续 N epoch 无提升则停止

# 重训练阶段
RETRAIN_EPOCHS =400        # 最优架构完整训练轮数
RETRAIN_PATIENCE = 15       # Early stopping patience

# 训练（CIFAR-10 上过大 batch 常不利于泛化；搜索阶段默认用更小 batch）
BATCH_SIZE = 1024
SEARCH_BATCH_SIZE = 1024       # 架构搜索时代理训练用；设为 None 则与 BATCH_SIZE 相同
INITIAL_LR = 0.1
WEIGHT_DECAY = 5e-4           # 略增可减轻过拟合，配合较小 batch
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.1         # 0 关闭；有助于泛化与小模型
WARMUP_EPOCHS = 5             # 线性 warmup 后再 cosine；0 表示仅 cosine
# 训练增强（仅 train 集）；0 关闭
RANDOM_ERASING_P = 0.25

# CIFAR-10 适配的 MobileNetV3 规模（通道数缩小）
STAGE_WIDTHS = [16, 24, 40, 80, 160]  # 5 个 stage 的输出通道
INPUT_CHANNEL = 16
LAST_CHANNEL = 256
NUM_CLASSES = 10
DROPOUT_RATE = 0.2
# True：搜索/训练使用 mobilenet_v3_dla.MobileNetV3CifarDLAHead（与 TensorRT DLA 导出一致）；False：GAP+Linear 版
USE_DLA_HEAD = True
CIFAR_IMAGE_SIZE = 32       # DLA 头用 dummy 前向推特征图尺寸，须与数据一致

# 搜索空间（对齐 MobileNetV3 维度，无 SE）
KS_LIST = [3, 5, 7]
EXPAND_RATIO_LIST = [3, 4, 5, 6, 7]
WIDTH_MULTIPLIER_LIST = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
DEPTH_MIN = 2
DEPTH_MAX = 4
# 须与 len(STAGE_WIDTHS) - 1 一致（首层 conv 后各 stage 的 block 堆叠数）
NUM_STAGES = 4

# 输出
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "nas_fbnet_outputs")
BEST_CONFIG_PATH = os.path.join(OUTPUT_DIR, "best_config.json")
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
# 若曾用旧版「以测试集为搜索目标」的 cache，与当前「验证集目标」混用会不一致，建议删 cache 后重搜
SEARCH_CACHE_PATH = os.path.join(OUTPUT_DIR, "search_cache.json")
SEARCH_CSV_PATH = os.path.join(OUTPUT_DIR, "search_log.csv")  # 列名含 val_acc / best_search_metric，与旧 CSV 不兼容时可删表头重跑
