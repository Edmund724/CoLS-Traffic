"""贝叶斯优化架构搜索"""
import csv
import json
import os
import shutil
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skopt import gp_minimize

from .config import (
    SEARCH_CALLS,
    SEARCH_EPOCHS,
    BATCH_SIZE,
    SEARCH_BATCH_SIZE,
    RANDOM_ERASING_P,
    OUTPUT_DIR,
    BEST_CONFIG_PATH,
    SEARCH_CACHE_PATH,
    SEARCH_CSV_PATH,
)
from .search_space import get_search_space, params_to_arch_config, arch_config_to_params
from .dataset import get_cifar10_loaders, get_cifar10_test_loader
from .train import train_model, evaluate, count_params, format_params, get_param_count

# search_log.csv 固定列顺序（新增列时：旧文件无新表头会先备份为 .bak 再写新表头）
_SEARCH_CSV_FIELDNAMES = (
    "search_number",
    "kernel_size",
    "expand_ratio",
    "width_multiplier",
    "depths",
    "total_params",
    "epochs_run",
    "epochs_max",
    "early_stopped",
    "test_acc",
    "best_test_acc",
    "best_search_metric",
    "val_acc",
    "model_path",
    "skipped",
)


def _max_test_acc_from_csv(csv_path):
    """从已有 search_log 的 test_acc 列恢复历史最大测试精度（续跑时累计 best_test_acc 用）。"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return None
    m = None
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "test_acc" not in reader.fieldnames:
                return None
            for row in reader:
                ta = (row.get("test_acc") or "").strip()
                if not ta:
                    continue
                try:
                    v = float(ta)
                    if m is None or v > m:
                        m = v
                except ValueError:
                    continue
    except OSError:
        return None
    return m


def _load_search_cache(cache_path, models_dir):
    """加载已评估配置：优先读 cache 文件，若无则从 trial_models 扫描恢复。"""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        evals = data.get("evals", [])
        # 兼容旧版 6 维 params：插入 width_idx=1 (1.0x) 到第 3 位
        for e in evals:
            p = e["params"]
            if len(p) == 6:
                e["params"] = [p[0], p[1], 1, p[2], p[3], p[4], p[5]]
        return evals

    # 从 trial_models 扫描 .pth 恢复（cache 文件不存在时）
    evals = []
    seen = set()
    if models_dir and os.path.isdir(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if not fname.endswith(".pth"):
                continue
            path = os.path.join(models_dir, fname)
            try:
                ckpt = torch.load(path, map_location="cpu")
                cfg = ckpt.get("config")
                acc = ckpt.get("test_acc")
                if cfg is not None and acc is not None:
                    params = arch_config_to_params(cfg)
                    key = tuple(params)
                    if key not in seen:
                        seen.add(key)
                        evals.append({"params": params, "result": -acc})
            except Exception:
                pass
    return evals


def _save_search_cache(cache_path, evals):
    """追加一条评估到 cache 文件。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"evals": evals}, f, indent=0)


def _append_search_csv(csv_path, row):
    """追加一行到搜索日志 CSV（列名固定，含 epochs_run / early_stopped）。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    nonempty = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if nonempty:
        with open(csv_path, encoding="utf-8") as f:
            first = f.readline()
        if "epochs_run" not in first or "best_test_acc" not in first:
            bak = csv_path + ".bak"
            if os.path.exists(bak):
                os.remove(bak)
            shutil.move(csv_path, bak)
            print(f"[search] 旧 search_log 已备份为 {bak}（表头需更新），新日志使用新表头。")

    full = {k: row.get(k, "") for k in _SEARCH_CSV_FIELDNAMES}
    need_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_SEARCH_CSV_FIELDNAMES))
        if need_header:
            writer.writeheader()
        writer.writerow(full)


def _plot_trial_curves(history, arch_config, trial_idx, save_dir):
    """绘制单次 trial 的训练/评估曲线并保存。"""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_series = [h.get("eval_acc") for h in history]
    test_series = [h.get("test_acc") for h in history]
    final_test = history[-1].get("final_test_acc") if history else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, train_loss, "b-", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    ax2.plot(epochs, train_acc, "g-", label="Train Acc")
    if any(v is not None for v in val_series):
        ax2.plot(epochs, val_series, "m-", label="Val Acc")
    if any(t is not None for t in test_series):
        ax2.plot(epochs, test_series, "r-", label="Test Acc (per epoch)")
    elif final_test is not None:
        ax2.axhline(final_test, color="r", linestyle="--", linewidth=1.5,
                    label=f"Test acc (once @ best val, {final_test:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    ax2.set_ylim(0, 1)

    fig.suptitle(f"Trial {trial_idx}  ks={arch_config['kernel_size']} "
                 f"expand={arch_config['expand_ratio']} w={arch_config.get('width_multiplier',1)} "
                 f"depths={arch_config['depths']}")
    fig.tight_layout()
    path = os.path.join(save_dir, f"trial_{trial_idx:03d}_curves.png")
    fig.savefig(path, dpi=100)
    plt.close()


def _make_objective(total_trials, curves_dir, models_dir, evals, cache_path, csv_path):
    """闭包：创建带 trial 计数和可视化的 objective。

    evals: 已评估列表 [{"params": [...], "result": -acc}, ...]，会被追加并持久化
    cache_path: 断点续跑 cache 文件路径
    csv_path: 搜索过程 CSV 日志路径
    """
    evaluated_cache = {tuple(e["params"]): e["result"] for e in evals}
    trial_counter = [len(evals)]  # 从已评估数继续，用于显示 Trial X/Total
    # 截至当前行：已完成 trial 中 test_acc（best-val 上的一次性测试精度）的最大值
    best_test_so_far = [_max_test_acc_from_csv(csv_path)]

    def _log_to_csv(
        trial_idx,
        arch_config,
        n_params,
        final_test_acc,
        best_val_acc,
        model_path,
        skipped=False,
        train_info=None,
    ):
        # result 存的是 -search_metric（验证集最优）；best 列为当前为止最优搜索指标
        rs = [e["result"] for e in evals]
        best_search_metric = max(-r for r in rs) if rs else 0.0
        depths_str = "-".join(str(x) for x in arch_config["depths"])
        if train_info is None:
            epochs_run_s = ""
            epochs_max_s = ""
            early_stopped_s = ""
        else:
            epochs_run_s = str(train_info["epochs_run"])
            epochs_max_s = str(train_info["epochs_max"])
            early_stopped_s = "true" if train_info["early_stopped"] else "false"
        if not skipped:
            try:
                t = float(final_test_acc)
                if best_test_so_far[0] is None or t > best_test_so_far[0]:
                    best_test_so_far[0] = t
            except (TypeError, ValueError):
                pass
        best_test_str = (
            f"{best_test_so_far[0]:.4f}" if best_test_so_far[0] is not None else ""
        )
        row = {
            "search_number": trial_idx,
            "kernel_size": arch_config["kernel_size"],
            "expand_ratio": arch_config["expand_ratio"],
            "width_multiplier": arch_config.get("width_multiplier", 1.0),
            "depths": depths_str,
            "total_params": n_params,
            "epochs_run": epochs_run_s,
            "epochs_max": epochs_max_s,
            "early_stopped": early_stopped_s,
            "test_acc": f"{final_test_acc:.4f}" if not skipped else "",
            "best_test_acc": best_test_str,
            "best_search_metric": f"{best_search_metric:.4f}",
            "val_acc": f"{best_val_acc:.4f}" if best_val_acc is not None else "",
            "model_path": model_path or "",
            "skipped": skipped,
        }
        _append_search_csv(csv_path, row)

    def objective(params):
        key = tuple(int(x) for x in params)
        if key in evaluated_cache:
            cached_val = evaluated_cache[key]
            trial_counter[0] += 1
            cfg = params_to_arch_config(params)
            n_params = get_param_count(cfg)
            m = -cached_val
            _log_to_csv(trial_counter[0], cfg, n_params, m, m, "", skipped=True, train_info=None)
            print(f"\n  [跳过] 配置已评估过: ks={cfg['kernel_size']} expand={cfg['expand_ratio']} "
                  f"w={cfg.get('width_multiplier',1)} depths={cfg['depths']} → search_metric={m:.4f}\n")
            return cached_val

        trial_counter[0] += 1
        trial_idx = trial_counter[0]
        arch_config = params_to_arch_config(params)

        n_params = get_param_count(arch_config)
        params_str = format_params(n_params)

        print(f"\n{'='*60}")
        print(f"  Trial {trial_idx}/{total_trials}")
        print(f"  架构: ks={arch_config['kernel_size']} expand={arch_config['expand_ratio']} "
              f"w={arch_config.get('width_multiplier',1)} depths={arch_config['depths']}")
        print(f"  参数量: {n_params} ({params_str})")
        print(f"{'='*60}")

        bs = SEARCH_BATCH_SIZE if SEARCH_BATCH_SIZE is not None else BATCH_SIZE
        train_loader, val_loader = get_cifar10_loaders(
            batch_size=bs,
            random_erasing_p=RANDOM_ERASING_P,
        )
        test_loader = get_cifar10_test_loader(batch_size=bs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ks, e, d = arch_config["kernel_size"], arch_config["expand_ratio"], arch_config["depths"]
        w = arch_config.get("width_multiplier", 1.0)
        d_str = "".join(str(x) for x in d)
        w_str = f"w{w}" if w != 1.0 else "w1"
        checkpoint_path = None
        if models_dir:
            checkpoint_path = os.path.join(
                models_dir,
                f"trial_{trial_idx:03d}_ks{ks}_e{e}_{w_str}_d{d_str}_{params_str}_best.pth",
            )

        # 早停与最优 checkpoint 基于验证集；贝叶斯目标为验证集最优（避免用测试集调参）
        best_metric, history, model, train_info = train_model(
            arch_config,
            train_loader,
            test_loader,
            epochs=SEARCH_EPOCHS,
            device=device,
            verbose=True,
            trial_info=(trial_idx, total_trials),
            val_loader=val_loader,
            checkpoint_path=checkpoint_path,
        )

        final_test_acc = history[-1].get("final_test_acc") if history else evaluate(model, test_loader, device)

        if curves_dir and history:
            _plot_trial_curves(history, arch_config, trial_idx, curves_dir)

        final_path = ""
        if models_dir and checkpoint_path:
            final_path = os.path.join(
                models_dir,
                f"trial_{trial_idx:03d}_ks{ks}_e{e}_{w_str}_d{d_str}_{params_str}_test{final_test_acc:.2f}.pth",
            )
            if checkpoint_path != final_path:
                os.rename(checkpoint_path, final_path)
            print(f"  模型已保存 → {final_path}")

        print(f"  Trial {trial_idx} 完成: val={best_metric:.4f}  test@best_val={final_test_acc:.4f}  "
              f"epochs={train_info['epochs_run']}/{train_info['epochs_max']}  "
              f"early_stopped={train_info['early_stopped']}\n")
        result = -best_metric
        evaluated_cache[key] = result
        evals.append({"params": list(key), "result": result})
        _save_search_cache(cache_path, evals)
        _log_to_csv(
            trial_idx,
            arch_config,
            n_params,
            final_test_acc,
            best_metric,
            os.path.abspath(final_path) if final_path else "",
            skipped=False,
            train_info=train_info,
        )
        return result

    return objective


def run_search():
    """运行贝叶斯优化搜索，保存最优配置。支持断点续跑。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    curves_dir = os.path.join(OUTPUT_DIR, "trial_curves")
    models_dir = os.path.join(OUTPUT_DIR, "trial_models")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    evals = _load_search_cache(SEARCH_CACHE_PATH, models_dir)
    x0 = [e["params"] for e in evals]
    y0 = [e["result"] for e in evals]

    space = get_search_space()
    objective_fn = _make_objective(SEARCH_CALLS, curves_dir, models_dir, evals, SEARCH_CACHE_PATH, SEARCH_CSV_PATH)

    n_remaining = SEARCH_CALLS - len(evals)
    print(f"\n搜索配置: 共 {SEARCH_CALLS} 次评估, 已完成 {len(evals)} 次, 剩余 {n_remaining} 次")
    print(f"每次 {SEARCH_EPOCHS} epochs, 曲线保存至: {curves_dir}")
    print(f"模型保存至: {models_dir}")
    print(f"断点 cache: {SEARCH_CACHE_PATH}")
    print(f"搜索日志 CSV: {SEARCH_CSV_PATH}\n")
    if evals:
        best_so_far = max(-r for r in y0)
        print(f"  续跑：当前最佳搜索指标（验证集）= {best_so_far:.4f}\n")

    # 不使用 CheckpointSaver：closure 无法 pickle，会导致 PicklingError
    gp_kwargs = dict(
        n_calls=SEARCH_CALLS,
        random_state=42,
        verbose=True,
    )
    if x0:
        gp_kwargs["x0"] = x0
        gp_kwargs["y0"] = y0
        gp_kwargs["n_initial_points"] = -len(x0)  # 负值表示 x0 已评估过，不重复
    result = gp_minimize(objective_fn, space, **gp_kwargs)

    best_params = result.x
    best_config = params_to_arch_config(best_params)
    best_val_metric = -result.fun

    with open(BEST_CONFIG_PATH, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n最优配置已保存 → {BEST_CONFIG_PATH}")
    print(f"搜索阶段最佳验证集准确率（贝叶斯目标）: {best_val_metric:.4f}")
    print("架构:", best_config)
    return best_config
