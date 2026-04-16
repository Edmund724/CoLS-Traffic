"""单架构训练逻辑"""
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR, SequentialLR
from tqdm import tqdm

from .models import MobileNetV3Cifar, MobileNetV3CifarDLAHead
from .config import (
    INITIAL_LR,
    WEIGHT_DECAY,
    MOMENTUM,
    STAGE_WIDTHS,
    INPUT_CHANNEL,
    LAST_CHANNEL,
    NUM_CLASSES,
    DROPOUT_RATE,
    SEARCH_PATIENCE,
    LABEL_SMOOTHING,
    WARMUP_EPOCHS,
    USE_DLA_HEAD,
    CIFAR_IMAGE_SIZE,
)


def logits_to_2d(out):
    """标准头为 (N,C)；DLA 头为 (N,C,1,1)，供 CrossEntropy / argmax 使用。"""
    if out.dim() == 4:
        return out.reshape(out.size(0), -1)
    return out


def build_model(arch_config):
    """按 config.USE_DLA_HEAD 构建搜索/训练用网络。"""
    common = dict(
        arch_config=arch_config,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        input_channel=INPUT_CHANNEL,
        stage_widths=STAGE_WIDTHS,
        last_channel=LAST_CHANNEL,
    )
    if USE_DLA_HEAD:
        return MobileNetV3CifarDLAHead(**common, image_size=CIFAR_IMAGE_SIZE)
    return MobileNetV3Cifar(**common)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = logits_to_2d(model(x))
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = logits_to_2d(model(x))
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total if total > 0 else 0.0


def count_params(model):
    """返回模型参数量。"""
    return sum(p.numel() for p in model.parameters())


def get_param_count(arch_config):
    """根据 arch_config 构建模型并返回参数量（用于训练前预估）。"""
    return count_params(build_model(arch_config))


def format_params(n):
    """格式化参数量，如 123456 -> 123k, 1234567 -> 1.2M"""
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{int(n/1e3)}k"
    return str(n)


def train_model(
    arch_config,
    train_loader,
    eval_loader,
    epochs,
    device=None,
    lr=None,
    use_cosine=True,
    verbose=False,
    trial_info=None,
    history_callback=None,
    patience=None,
    val_loader=None,
    checkpoint_path=None,
    label_smoothing=None,
    warmup_epochs=None,
):
    """训练指定架构。

    若提供 val_loader：早停与最优 checkpoint 基于 **验证集**（避免用测试集调参），
    训练结束后在最佳验证权重上计算 test_acc，并作为返回值之一。
    若未提供 val_loader：行为与旧版一致，早停与最优 checkpoint 基于 eval_loader（通常为测试集）。

    返回:
        (best_metric_acc, history, model, train_info): train_info 含 epochs_run、early_stopped、epochs_max。
        best_metric_acc 为验证集最优（有 val 时）或 eval_loader 上最优。
        若提供了 val_loader：训练循环内**不**评估测试集（避免反复看 test）；history 中 test_acc 为 None，
        仅在结束后写入 history[-1]["final_test_acc"]。最终 model 为验证集最优权重。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = lr or INITIAL_LR
    ls = LABEL_SMOOTHING if label_smoothing is None else label_smoothing
    wu = WARMUP_EPOCHS if warmup_epochs is None else warmup_epochs

    model = build_model(arch_config).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = None
    if use_cosine and wu > 0 and epochs > wu:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=wu),
                CosineAnnealingLR(optimizer, T_max=max(1, epochs - wu)),
            ],
            milestones=[wu],
        )
    elif use_cosine and wu > 0 and epochs <= wu:
        scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=epochs)
    elif use_cosine:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[epochs * 3 // 4], gamma=0.1)

    use_val_for_best = val_loader is not None
    best_metric = -1.0
    best_state = None
    history = []
    no_improve = 0
    patience = patience if patience is not None else SEARCH_PATIENCE
    pbar = tqdm(range(epochs), desc="Epoch", disable=not verbose)

    for ep in pbar:
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 有验证集时不在 epoch 内评估测试集，避免训练过程中反复「看」test
        if use_val_for_best:
            test_acc = None
        else:
            test_acc = evaluate(model, eval_loader, device)
        val_acc = evaluate(model, val_loader, device) if val_loader is not None else None
        if scheduler is not None:
            scheduler.step()

        if use_val_for_best:
            metric = val_acc if val_acc is not None else 0.0
        else:
            metric = test_acc

        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if checkpoint_path:
                n_params = count_params(model)
                torch.save({
                    "config": arch_config,
                    "state_dict": best_state,
                    "test_acc": test_acc,
                    "val_acc": val_acc if val_acc is not None else None,
                    "n_params": n_params,
                    "use_dla_head": USE_DLA_HEAD,
                }, checkpoint_path)
        else:
            no_improve += 1

        record = {"epoch": ep + 1, "train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}
        if val_acc is not None:
            record["eval_acc"] = val_acc
        history.append(record)
        if history_callback:
            history_callback(record)
        if verbose:
            parts = [f"loss={train_loss:.4f}", f"train_acc={train_acc:.4f}"]
            if val_acc is not None:
                parts.append(f"val_acc={val_acc:.4f}")
            if not use_val_for_best:
                parts.append(f"test_acc={test_acc:.4f}")
            parts.append(f"best_metric={best_metric:.4f}")
            pbar.set_postfix_str(", ".join(parts))
        if patience > 0 and no_improve >= patience:
            if verbose:
                pbar.set_postfix_str(f"Early stop @ epoch {ep + 1}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    # 若按验证集选权，此处得到的是验证最优权重上的测试集精度（供日志与汇报）
    final_test_acc = evaluate(model, eval_loader, device)
    if history:
        history[-1]["final_test_acc"] = final_test_acc
    epochs_run = len(history)
    train_info = {
        "epochs_run": epochs_run,
        "epochs_max": epochs,
        "early_stopped": epochs_run < epochs,
    }
    return best_metric, history, model, train_info
