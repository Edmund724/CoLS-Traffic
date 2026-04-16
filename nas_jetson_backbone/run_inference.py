#!/usr/bin/env python3
"""NAS-FBNet 推理脚本

模型文件内嵌 config，只需指定 --model 即可。

用法:
  python run_inference.py                          # 使用默认 best_model.pth
  python run_inference.py --model xxx.pth          # 指定任意模型文件
  python run_inference.py --model xxx.pth --image a.png  # 单张图片推理
"""
import argparse
import json
import os
import sys
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_fbnet.models import MobileNetV3Cifar
from nas_fbnet.config import (
    STAGE_WIDTHS,
    INPUT_CHANNEL,
    LAST_CHANNEL,
    NUM_CLASSES,
    BEST_CONFIG_PATH,
    BEST_MODEL_PATH,
    CIFAR10_ROOT,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 标准化
NORMALIZE = transforms.Normalize(
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010),
)


def load_model(model_path, device=None):
    """从模型文件加载（文件内嵌 config + state_dict）。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "config" in ckpt:
        arch_config = {k: v for k, v in ckpt["config"].items() if k in ["kernel_size", "expand_ratio", "depths"]}
        state_dict = ckpt["state_dict"]
    else:
        # 兼容旧格式：仅 state_dict，需从 best_config.json 读 config
        if not os.path.exists(BEST_CONFIG_PATH):
            raise FileNotFoundError(f"模型为旧格式(仅权重)，需提供 config。请使用 best_config.json 或重新保存模型。")
        with open(BEST_CONFIG_PATH) as f:
            cfg = json.load(f)
        arch_config = {k: v for k, v in cfg.items() if k in ["kernel_size", "expand_ratio", "depths"]}
        state_dict = ckpt

    model = MobileNetV3Cifar(
        arch_config,
        num_classes=NUM_CLASSES,
        dropout_rate=0.0,
        input_channel=INPUT_CHANNEL,
        stage_widths=STAGE_WIDTHS,
        last_channel=LAST_CHANNEL,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, arch_config


def evaluate_test_set(model, data_root, device, batch_size=128):
    """在 CIFAR-10 测试集上评估。"""
    from torch.utils.data import DataLoader
    from torchvision import datasets

    data_dir = os.path.dirname(data_root) if os.path.basename(data_root) == "cifar-10-batches-py" else data_root
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALIZE,
    ])
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def predict_image(model, image_path, device):
    """对单张图片进行预测。"""
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        NORMALIZE,
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1).item()
    return pred, probs[0].cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description="NAS-FBNet 推理")
    parser.add_argument("--model", default=BEST_MODEL_PATH, help="模型 .pth 路径（内嵌 config）")
    parser.add_argument("--image", type=str, help="单张图片路径（可选）")
    parser.add_argument("--batch-size", type=int, default=128, help="测试集 batch size")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"未找到模型: {args.model}")
        print("请先运行 python run_search.py 或 python run_retrain.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, arch_config = load_model(args.model, device)
    print(f"已加载模型: {args.model}")
    print(f"架构: {arch_config}\n")

    if args.image:
        if not os.path.exists(args.image):
            print(f"图片不存在: {args.image}")
            sys.exit(1)
        pred, probs = predict_image(model, args.image, device)
        print(f"预测类别: {CIFAR10_CLASSES[pred]} (idx={pred})")
        print("各类别概率:")
        for i, (name, p) in enumerate(zip(CIFAR10_CLASSES, probs)):
            print(f"  {name}: {p:.4f}")
    else:
        acc = evaluate_test_set(model, CIFAR10_ROOT, device, args.batch_size)
        print(f"CIFAR-10 测试集准确率: {acc:.4f}")


if __name__ == "__main__":
    main()
