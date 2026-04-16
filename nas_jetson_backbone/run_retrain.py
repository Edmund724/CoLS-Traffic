#!/usr/bin/env python3
"""对搜索得到的最优架构进行完整重训练"""
import json
import sys
import os
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_fbnet.models import MobileNetV3Cifar
from nas_fbnet.dataset import get_cifar10_loaders, get_cifar10_test_loader
from nas_fbnet.train import train_epoch, evaluate
from nas_fbnet.config import (
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
    BEST_CONFIG_PATH,
    BEST_MODEL_PATH,
    OUTPUT_DIR,
)


def main():
    if not os.path.exists(BEST_CONFIG_PATH):
        print(f"未找到最优配置，请先运行: python run_search.py")
        print(f"  期望路径: {BEST_CONFIG_PATH}")
        return

    with open(BEST_CONFIG_PATH) as f:
        arch_config = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileNetV3Cifar(
        arch_config,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        input_channel=INPUT_CHANNEL,
        stage_widths=STAGE_WIDTHS,
        last_channel=LAST_CHANNEL,
    ).to(device)

    train_loader, val_loader = get_cifar10_loaders()
    test_loader = get_cifar10_test_loader()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=RETRAIN_EPOCHS)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"重训练最优架构 (共 {RETRAIN_EPOCHS} epochs)")
    print(f"架构: {arch_config}\n")

    for epoch in range(RETRAIN_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "config": arch_config,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, BEST_MODEL_PATH)
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1}/{RETRAIN_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  best={best_val_acc:.4f}@{best_epoch}")

        if patience_counter >= RETRAIN_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    test_acc = evaluate(model, test_loader, device)
    print(f"\n{'='*50}")
    print(f"重训练完成")
    print(f"最佳验证准确率: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"测试集准确率:   {test_acc:.4f}")
    print(f"模型已保存 → {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
