#!/usr/bin/env python3
"""使用 ONNX Runtime 对 NAS-FBNet 导出的模型推理（与 run_inference 预处理一致）

依赖: pip install onnxruntime  （GPU: pip install onnxruntime-gpu）

用法:
  python run_inference_onnx.py --model nas_fbnet_outputs/onnx_models/xxx.onnx
  python run_inference_onnx.py --model xxx.onnx --image a.png
  python run_inference_onnx.py   # 若 onnx_models 下仅有一个 .onnx 则自动选用
"""
import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_fbnet.config import CIFAR10_ROOT, OUTPUT_DIR

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)


def _get_session(model_path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("请先安装: pip install onnxruntime")
        sys.exit(1)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(model_path, providers=providers)
    except Exception:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return sess


def _preprocess_numpy_chw(images_nchw):
    """images: float32 NCHW [0,1]，与 torchvision Normalize 一致"""
    return (images_nchw - MEAN) / STD


def _predict_batch(sess, x_np):
    """x_np: NCHW float32，已归一化"""
    input_name = sess.get_inputs()[0].name
    out = sess.run(None, {input_name: x_np})[0]
    return out


def evaluate_test_set(sess, data_root, batch_size=128):
    """CIFAR-10 测试集准确率"""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    data_dir = os.path.dirname(data_root) if os.path.basename(data_root) == "cifar-10-batches-py" else data_root
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    correct, total = 0, 0
    for x, y in tqdm(loader, desc="CIFAR-10 test", unit="batch", leave=True):
        x_np = x.numpy().astype(np.float32)
        x_np = _preprocess_numpy_chw(x_np)
        logits = _predict_batch(sess, x_np)
        pred = np.argmax(logits, axis=1)
        y_np = y.numpy()
        correct += (pred == y_np).sum()
        total += y_np.size
    return correct / total if total > 0 else 0.0


def predict_image(sess, image_path):
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).numpy().astype(np.float32)
    x = _preprocess_numpy_chw(x)
    logits = _predict_batch(sess, x)[0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    pred = int(np.argmax(logits))
    return pred, probs.tolist()


def _resolve_default_onnx():
    onnx_dir = os.path.join(OUTPUT_DIR, "onnx_models")
    if not os.path.isdir(onnx_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(onnx_dir, "*.onnx")))
    return candidates[0] if len(candidates) == 1 else None


def main():
    parser = argparse.ArgumentParser(description="NAS-FBNet ONNX 推理")
    parser.add_argument("--model", type=str, default='./nas_fbnet_outputs/onnx/trial_012_ks3_e6_w1.25_d4344_2.9M_test0.86.onnx', help="ONNX 模型路径")
    parser.add_argument("--image", type=str, help="单张图片路径（可选）")
    parser.add_argument("--batch-size", type=int, default=1, help="测试集 batch size")
    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        model_path = _resolve_default_onnx()
        if not model_path:
            print("请指定 --model 或在 nas_fbnet_outputs/onnx_models/ 下只放一个 .onnx 以自动选择")
            sys.exit(1)
        print(f"使用: {model_path}")

    if not os.path.isfile(model_path):
        print(f"未找到 ONNX 文件: {model_path}")
        sys.exit(1)

    print(f"加载 ONNX: {model_path}")
    sess = _get_session(model_path)
    providers = sess.get_providers()
    print(f"执行 providers: {providers}\n")

    if args.image:
        if not os.path.exists(args.image):
            print(f"图片不存在: {args.image}")
            sys.exit(1)
        pred, probs = predict_image(sess, args.image)
        print(f"预测类别: {CIFAR10_CLASSES[pred]} (idx={pred})")
        print("各类别概率:")
        for name, p in zip(CIFAR10_CLASSES, probs):
            print(f"  {name}: {p:.4f}")
    else:
        acc = evaluate_test_set(sess, CIFAR10_ROOT, args.batch_size)
        print(f"CIFAR-10 测试集准确率: {acc:.4f}")


if __name__ == "__main__":
    main()
