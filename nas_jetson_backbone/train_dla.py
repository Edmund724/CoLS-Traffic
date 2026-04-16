#!/usr/bin/env python3
"""固定架构训练（不参与 NAS）：使用 DLA 友好分类头，与搜索代码解耦。

- 骨干与 arch_config 与 MobileNetV3Cifar 一致（见 nas_fbnet/models/mobilenet_v3.py）。
- 分类头改为「1×1 Conv + 固定核 AvgPool2d + 1×1 Conv」，避免 GAP(GlobalAveragePool) +
  全连接 + Flatten 在 TensorRT DLA 上落到 GPU 的问题；便于整网尽量跑在 DLA。

不修改 nas_fbnet/train.py / search.py；本脚本内自建训练循环，仅复用 train_epoch / evaluate 等工具函数。

用法:
  python train_dla.py
  python train_dla.py --epochs 300 --patience 20 --image-size 32
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_fbnet.dataset import get_cifar10_loaders, get_cifar10_test_loader
from nas_fbnet.models.mobilenet_v3_dla import MobileNetV3CifarDLAHead
from nas_fbnet.train import count_params, format_params
from nas_fbnet.config import (
    OUTPUT_DIR,
    RETRAIN_EPOCHS,
    RETRAIN_PATIENCE,
    INITIAL_LR,
    WEIGHT_DECAY,
    MOMENTUM,
    STAGE_WIDTHS,
    INPUT_CHANNEL,
    LAST_CHANNEL,
    NUM_CLASSES,
    DROPOUT_RATE,
    SEARCH_PATIENCE,
)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# 与 search_space 上界一致的一组；可自行修改
FIXED_ARCH_CONFIG = {
    "kernel_size": 3,
    "expand_ratio": 6,
    "width_multiplier": 1.25,
    "depths": [4, 4, 2, 2],
}


def _dla_logits_to_2d(out):
    """DLA 头前向为 (N, C, 1, 1)，供 CrossEntropyLoss / argmax 需 (N, C)。"""
    if out.dim() == 4:
        return out.squeeze(-1).squeeze(-1)
    return out


def train_epoch_dla(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        out = _dla_logits_to_2d(out)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate_dla(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        out = _dla_logits_to_2d(out)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def build_model(arch_config, image_size):
    return MobileNetV3CifarDLAHead(
        arch_config,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        input_channel=INPUT_CHANNEL,
        stage_widths=STAGE_WIDTHS,
        last_channel=LAST_CHANNEL,
        image_size=image_size,
    )


def train_fixed_dla(
    arch_config,
    train_loader,
    test_loader,
    epochs,
    device,
    image_size=32,
    val_loader=None,
    patience=None,
    checkpoint_path=None,
    verbose=True,
):
    """训练并在 **验证集** 上早停、保存最优 checkpoint；训练结束后在 **测试集** 上评一次。

    若 ``val_loader`` 为 None，则退化为用测试集做早停（不推荐）。
    """
    lr = INITIAL_LR
    model = build_model(arch_config, image_size=image_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_val = val_loader is not None
    best_metric = 0.0
    test_acc_at_best = 0.0
    best_state = None
    history = []
    no_improve = 0
    patience = patience if patience is not None else SEARCH_PATIENCE
    pbar = tqdm(range(epochs), desc="Epoch", disable=not verbose)

    for ep in pbar:
        train_loss, train_acc = train_epoch_dla(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate_dla(model, val_loader, device) if use_val else None
        test_acc = evaluate_dla(model, test_loader, device)
        scheduler.step()

        monitor = val_acc if use_val else test_acc
        if monitor > best_metric:
            best_metric = monitor
            test_acc_at_best = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if checkpoint_path:
                n_params = count_params(model)
                torch.save(
                    {
                        "config": arch_config,
                        "model_type": "MobileNetV3CifarDLAHead",
                        "image_size": image_size,
                        "logits_4d": True,
                        "early_stop_on": "val" if use_val else "test",
                        "best_val_acc": val_acc if use_val else None,
                        "test_acc_at_best_epoch": test_acc,
                        "state_dict": best_state,
                        "n_params": n_params,
                    },
                    checkpoint_path,
                )
        else:
            no_improve += 1

        record = {
            "epoch": ep + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
        }
        if val_acc is not None:
            record["val_acc"] = val_acc
        history.append(record)
        if verbose:
            parts = [f"loss={train_loss:.4f}", f"train_acc={train_acc:.4f}"]
            if val_acc is not None:
                parts.append(f"val_acc={val_acc:.4f}")
            parts.append(f"test_acc={test_acc:.4f}")
            parts.append(f"best_val={best_metric:.4f}" if use_val else f"best_test={best_metric:.4f}")
            pbar.set_postfix_str(", ".join(parts))
        if patience > 0 and no_improve >= patience:
            if verbose:
                pbar.set_postfix_str(f"Early stop @ epoch {ep + 1} (monitor={'val' if use_val else 'test'})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    final_test_acc = evaluate_dla(model, test_loader, device)
    return {
        "test_acc": final_test_acc,
        "best_val_acc": best_metric if use_val else None,
        "test_acc_at_best_val_epoch": test_acc_at_best if use_val else None,
        "history": history,
        "model": model,
    }


def main():
    parser = argparse.ArgumentParser(description="固定大模型（DLA 头）CIFAR-10 训练")
    parser.add_argument("--epochs", type=int, default=RETRAIN_EPOCHS, help="训练轮数")
    parser.add_argument(
        "--patience",
        type=int,
        default=RETRAIN_PATIENCE,
        help="早停：验证集 val_acc 连续无提升的 epoch 数（0 关闭；无 val 时退化为 test）",
    )
    parser.add_argument("--output", type=str, default=None, help="保存 .pth 路径（默认 OUTPUT_DIR 下自动生成）")
    parser.add_argument(
        "--image-size",
        type=int,
        default=32,
        help="CIFAR 输入边长；用于推断 final_expand 后空间尺寸以构建固定 AvgPool（默认 32）",
    )
    args = parser.parse_args()

    arch_config = dict(FIXED_ARCH_CONFIG)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = args.image_size
    model_for_count = build_model(arch_config, image_size=image_size)
    n_params = count_params(model_for_count)
    params_str = format_params(n_params)
    d_str = "".join(str(d) for d in arch_config["depths"])
    base_name = (
        f"fixed_dla_ks{arch_config['kernel_size']}_e{arch_config['expand_ratio']}"
        f"_w{arch_config['width_multiplier']}_d{d_str}_{params_str}"
    )
    if args.output:
        stem = os.path.splitext(args.output)[0]
        checkpoint_path = stem + "_best.pth"
    else:
        checkpoint_path = os.path.join(OUTPUT_DIR, base_name + "_best.pth")

    print("固定架构训练（DLA 友好头，与 NAS 搜索模型导出分离）")
    print(f"  arch_config: {arch_config}")
    print(f"  image_size: {image_size}")
    print(f"  参数量: {n_params} ({params_str})")
    print(f"  epochs={args.epochs}, patience={args.patience}")
    print(f"  训练中 checkpoint: {checkpoint_path}\n")

    train_loader, val_loader = get_cifar10_loaders()
    test_loader = get_cifar10_test_loader()

    result = train_fixed_dla(
        arch_config,
        train_loader,
        test_loader,
        epochs=args.epochs,
        device=device,
        image_size=image_size,
        val_loader=val_loader,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    test_acc = result["test_acc"]
    model = result["model"]

    if args.output:
        stem = os.path.splitext(args.output)[0]
        out_final = stem + f"_test{test_acc:.2f}.pth"
    else:
        out_final = os.path.join(OUTPUT_DIR, f"{base_name}_test{test_acc:.2f}.pth")

    if os.path.isfile(checkpoint_path) and os.path.abspath(checkpoint_path) != os.path.abspath(out_final):
        os.rename(checkpoint_path, out_final)
    else:
        torch.save(
            {
                "config": arch_config,
                "model_type": "MobileNetV3CifarDLAHead",
                "image_size": image_size,
                "logits_4d": True,
                "early_stop_on": "val" if val_loader is not None else "test",
                "best_val_acc": result["best_val_acc"],
                "test_acc_at_best_val_epoch": result["test_acc_at_best_val_epoch"],
                "state_dict": model.state_dict(),
                "test_acc": test_acc,
                "n_params": n_params,
            },
            out_final,
        )

    cfg_path = os.path.join(OUTPUT_DIR, "fixed_large_dla_config.json")
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "arch_config": arch_config,
                "model_type": "MobileNetV3CifarDLAHead",
                "image_size": image_size,
                "logits_4d": True,
            },
            f,
            indent=2,
        )

    bv = result["best_val_acc"]
    if bv is not None:
        print(
            f"\n训练结束  best_val_acc={bv:.4f}  "
            f"test@that_epoch={result['test_acc_at_best_val_epoch']:.4f}  "
            f"final_test_acc={test_acc:.4f}"
        )
    else:
        print(f"\n训练结束  test_acc={test_acc:.4f}")
    print(f"模型 → {out_final}")
    print(f"配置 → {cfg_path}")


if __name__ == "__main__":
    main()
