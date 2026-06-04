"""
硬件反馈架构搜索（独立实现，不修改 search.py）。

流程对齐 tf_nas_fpga/nas_train_qkeras_HW.py：
  1) 用 XGBoost 代理预测延迟/功耗并归一化；
  2) 可选：预测延迟超阈值则跳过训练；
  3) 训练后组合目标（最小化）：
       combined = HW_ALPHA * (1 - val_acc) + HW_BETA * latency_norm + HW_GAMMA * power_norm
  4) skopt 最小化 combined（与原版「最小化 -val_acc」不同，请使用独立 search_cache_hw.json）。

运行：项目根目录  python run_search_hw.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skopt import gp_minimize

from nas_fbnet.config_hw import (
    BATCH_SIZE,
    BEST_CONFIG_PATH,
    CIFAR10_ROOT,
    HW_ALPHA,
    HW_BETA,
    HW_GAMMA,
    HW_LATENCY_MAX_MS,
    HW_LATENCY_MIN_MS,
    HW_MAX_PRED_LATENCY_MS,
    HW_POWER_MAX_W,
    HW_POWER_MIN_W,
    HW_PROXY_MODEL_DIR,
    HW_SKIP_LOSS,
    OUTPUT_DIR,
    RANDOM_ERASING_P,
    SEARCH_SAVE_PLOTS,
    SEARCH_BATCH_SIZE,
    SEARCH_CACHE_PATH,
    SEARCH_CALLS,
    SEARCH_CSV_PATH,
    SEARCH_EPOCHS,
    STAGE_WIDTHS,
    INPUT_CHANNEL,
    LAST_CHANNEL,
)
from nas_fbnet.hw_proxy_jetson import JetsonHWProxy
from nas_fbnet.search_space_hw import arch_config_to_params, get_search_space, params_to_arch_config
from nas_fbnet.checkpoint_naming import infer_filename_suffix
from nas_fbnet.dataset import get_cifar10_loaders, get_cifar10_test_loader
from nas_fbnet.models.mbconv import make_divisible
from nas_fbnet.train_hw import evaluate, format_params, get_param_count, train_model

# 在原 search_log 列基础上增加硬件相关列
_SEARCH_HW_FIELDNAMES = (
    "search_number",
    "kernel_size",
    "expand_ratio",
    "width_multiplier",
    "depths",
    "stem_c",
    "stage1_c",
    "stage2_c",
    "stage3_c",
    "stage4_c",
    "last_c",
    "infer_mode",
    "quant_precision",
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
    "hw_alpha",
    "hw_beta",
    "hw_gamma",
    "pred_latency_ms",
    "pred_power_w",
    "latency_norm",
    "power_norm",
    "combined_loss",
    "best_combined",
    "trial_elapsed_sec",
    "time_taken_sec",
    "search_elapsed_sec",
    "run_elapsed_sec",
    "hw_skip_reason",
)


def _derive_channel_signature(arch_config: dict) -> dict[str, int]:
    w = float(arch_config.get("width_multiplier", 1.0))
    stage_widths = [make_divisible(c * w) for c in STAGE_WIDTHS]
    return {
        "stem_c": make_divisible(INPUT_CHANNEL * w),
        "stage1_c": stage_widths[1],
        "stage2_c": stage_widths[2],
        "stage3_c": stage_widths[3],
        "stage4_c": stage_widths[4],
        "last_c": make_divisible(LAST_CHANNEL * w),
    }


def _max_test_acc_from_csv(csv_path: str):
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


def _max_val_acc_from_csv(csv_path: str) -> float:
    """续跑时从 search_log_hw.csv 恢复历史最大验证集精度。"""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return 0.0
    m = 0.0
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "val_acc" not in reader.fieldnames:
                return 0.0
            for row in reader:
                va = (row.get("val_acc") or "").strip()
                if not va:
                    continue
                try:
                    v = float(va)
                    if v > m:
                        m = v
                except ValueError:
                    continue
    except OSError:
        return 0.0
    return m


def _load_hw_cache(cache_path: str) -> list:
    """仅读 JSON；不从 trial_models 扫描（扫描结果与 combined 目标不一致）。"""
    if not os.path.exists(cache_path):
        return []
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    evals = data.get("evals", [])
    for e in evals:
        p = e["params"]
        if len(p) == 6:
            e["params"] = [p[0], p[1], 1, p[2], p[3], p[4], p[5]]
        if len(p) == 7:
            e["params"] = list(p) + [0, 0]
    return evals


def _save_hw_cache(cache_path: str, evals: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"evals": evals}, f, indent=0)


def _append_hw_csv(csv_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    full = {k: row.get(k, "") for k in _SEARCH_HW_FIELDNAMES}
    need_header = not (os.path.exists(csv_path) and os.path.getsize(csv_path) > 0)
    nonempty = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    if nonempty:
        with open(csv_path, encoding="utf-8") as f:
            first = f.readline()
        if (
            "combined_loss" not in first
            or "trial_elapsed_sec" not in first
            or "run_elapsed_sec" not in first
            or "stem_c" not in first
        ):
            bak = csv_path + ".bak"
            if os.path.exists(bak):
                os.remove(bak)
            shutil.move(csv_path, bak)
            print(f"[search_hw] 旧 CSV 已备份为 {bak}，使用新表头。")
            need_header = True
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(_SEARCH_HW_FIELDNAMES))
        if need_header:
            w.writeheader()
        w.writerow(full)


def _plot_trial_curves(history, arch_config, trial_idx, save_dir):
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
        ax2.axhline(
            final_test,
            color="r",
            linestyle="--",
            linewidth=1.5,
            label=f"Test acc (once @ best val, {final_test:.4f})",
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    ax2.set_ylim(0, 1)

    fig.suptitle(
        f"[HW] Trial {trial_idx}  ks={arch_config['kernel_size']} "
        f"expand={arch_config['expand_ratio']} w={arch_config.get('width_multiplier', 1)} "
        f"depths={arch_config['depths']}"
    )
    fig.tight_layout()
    path = os.path.join(save_dir, f"trial_{trial_idx:03d}_curves.png")
    fig.savefig(path, dpi=100)
    plt.close()


def _combined_loss(val_acc: float, lat_n: float, pow_n: float) -> float:
    return (
        HW_ALPHA * (1.0 - val_acc)
        + HW_BETA * lat_n
        + HW_GAMMA * pow_n
    )


def _make_objective_hw(
    total_trials: int,
    curves_dir: str,
    models_dir: str,
    evals: list,
    cache_path: str,
    csv_path: str,
    proxy: JetsonHWProxy,
    run_start_time: float,
    train_loader,
    val_loader,
    test_loader,
    device,
):
    evaluated_cache = {tuple(int(x) for x in e["params"]): e["result"] for e in evals}
    trial_counter = [len(evals)]
    best_test_so_far = [_max_test_acc_from_csv(csv_path)]
    valid_comb = [e["result"] for e in evals if e["result"] < HW_SKIP_LOSS - 1]
    best_combined_so_far = [min(valid_comb) if valid_comb else float("inf")]
    best_val_metric_so_far = [_max_val_acc_from_csv(csv_path)]
    search_start_time = time.perf_counter()

    def _log_row2(
        trial_idx,
        arch_config,
        n_params,
        final_test_acc,
        best_val_acc,
        model_path,
        *,
        skipped=False,
        train_info=None,
        hw=None,
        combined=None,
        hw_skip_reason="",
        trial_elapsed_sec=None,
        time_taken_sec=None,
        search_elapsed_sec=None,
        run_elapsed_sec=None,
    ):
        depths_str = "-".join(str(x) for x in arch_config["depths"])
        if train_info is None:
            epochs_run_s = epochs_max_s = early_stopped_s = ""
        else:
            epochs_run_s = str(train_info["epochs_run"])
            epochs_max_s = str(train_info["epochs_max"])
            early_stopped_s = "true" if train_info["early_stopped"] else "false"

        if not skipped and best_val_acc is not None:
            try:
                v = float(best_val_acc)
                if v > best_val_metric_so_far[0]:
                    best_val_metric_so_far[0] = v
            except (TypeError, ValueError):
                pass

        if not skipped and final_test_acc is not None:
            try:
                t = float(final_test_acc)
                if best_test_so_far[0] is None or t > best_test_so_far[0]:
                    best_test_so_far[0] = t
            except (TypeError, ValueError):
                pass

        if combined is not None and combined < HW_SKIP_LOSS - 1:
            if combined < best_combined_so_far[0]:
                best_combined_so_far[0] = combined

        best_test_str = (
            f"{best_test_so_far[0]:.4f}" if best_test_so_far[0] is not None else ""
        )
        ch = _derive_channel_signature(arch_config)
        row = {
            "search_number": trial_idx,
            "kernel_size": arch_config["kernel_size"],
            "expand_ratio": arch_config["expand_ratio"],
            "width_multiplier": arch_config.get("width_multiplier", 1.0),
            "depths": depths_str,
            "stem_c": ch["stem_c"],
            "stage1_c": ch["stage1_c"],
            "stage2_c": ch["stage2_c"],
            "stage3_c": ch["stage3_c"],
            "stage4_c": ch["stage4_c"],
            "last_c": ch["last_c"],
            "infer_mode": arch_config.get("infer_mode", ""),
            "quant_precision": arch_config.get("quant_precision", ""),
            "total_params": n_params,
            "epochs_run": epochs_run_s,
            "epochs_max": epochs_max_s,
            "early_stopped": early_stopped_s,
            "test_acc": f"{final_test_acc:.4f}"
            if not skipped and final_test_acc is not None
            else "",
            "best_test_acc": best_test_str,
            "best_search_metric": f"{best_val_metric_so_far[0]:.4f}",
            "val_acc": f"{best_val_acc:.4f}" if best_val_acc is not None else "",
            "model_path": model_path or "",
            "skipped": skipped,
            "hw_alpha": HW_ALPHA,
            "hw_beta": HW_BETA,
            "hw_gamma": HW_GAMMA,
            "pred_latency_ms": f"{hw['pred_latency_ms']:.6f}" if hw else "",
            "pred_power_w": f"{hw['pred_power_w']:.6f}" if hw else "",
            "latency_norm": f"{hw['latency_norm']:.6f}" if hw else "",
            "power_norm": f"{hw['power_norm']:.6f}" if hw else "",
            "combined_loss": f"{combined:.6f}" if combined is not None else "",
            "best_combined": f"{best_combined_so_far[0]:.6f}"
            if best_combined_so_far[0] < HW_SKIP_LOSS - 1
            else "",
            "trial_elapsed_sec": f"{float(trial_elapsed_sec):.6f}" if trial_elapsed_sec is not None else "",
            "time_taken_sec": f"{float(time_taken_sec):.6f}" if time_taken_sec is not None else "",
            "search_elapsed_sec": f"{float(search_elapsed_sec):.6f}" if search_elapsed_sec is not None else "",
            "run_elapsed_sec": f"{float(run_elapsed_sec):.6f}" if run_elapsed_sec is not None else "",
            "hw_skip_reason": hw_skip_reason,
        }
        _append_hw_csv(csv_path, row)

    def objective(params):
        trial_start_time = time.perf_counter()
        key = tuple(int(x) for x in params)
        if key in evaluated_cache:
            trial_counter[0] += 1
            cfg = params_to_arch_config(params)
            n_params = get_param_count(cfg)
            c = evaluated_cache[key]
            trial_elapsed = time.perf_counter() - trial_start_time
            search_elapsed = time.perf_counter() - search_start_time
            run_elapsed = time.perf_counter() - run_start_time
            _log_row2(
                trial_counter[0],
                cfg,
                n_params,
                None,
                None,
                "",
                skipped=True,
                train_info=None,
                hw=None,
                combined=c,
                hw_skip_reason="cache_hit",
                trial_elapsed_sec=trial_elapsed,
                time_taken_sec=trial_elapsed,
                search_elapsed_sec=search_elapsed,
                run_elapsed_sec=run_elapsed,
            )
            print(
                f"\n  [HW 跳过-cache] 配置已评估 combined={c:.6f}: "
                f"ks={cfg['kernel_size']} … infer={cfg.get('infer_mode')} "
                f"time={trial_elapsed:.3f}s search_total={search_elapsed:.1f}s run_total={run_elapsed:.1f}s\n"
            )
            return c

        trial_counter[0] += 1
        trial_idx = trial_counter[0]
        arch_config = params_to_arch_config(params)
        n_params = get_param_count(arch_config)
        params_str = format_params(n_params)
        ch = _derive_channel_signature(arch_config)
        ch_str = (
            f"stem={ch['stem_c']} s1={ch['stage1_c']} s2={ch['stage2_c']} "
            f"s3={ch['stage3_c']} s4={ch['stage4_c']} last={ch['last_c']}"
        )

        hw_pred = proxy.predict_normalized(
            arch_config,
            n_params,
            lat_min_ms=HW_LATENCY_MIN_MS,
            lat_max_ms=HW_LATENCY_MAX_MS,
            pwr_min_w=HW_POWER_MIN_W,
            pwr_max_w=HW_POWER_MAX_W,
            require_power=HW_GAMMA > 0.0,
        )

        print(f"\n{'='*60}")
        print(f"  [HW] Trial {trial_idx}/{total_trials}")
        print(
            f"  架构: ks={arch_config['kernel_size']} expand={arch_config['expand_ratio']} "
            f"w={arch_config.get('width_multiplier', 1)} depths={arch_config['depths']} "
            f"infer={arch_config.get('infer_mode')} quant={arch_config.get('quant_precision')}"
        )
        print(f"  通道: {ch_str}")
        print(f"  参数量: {n_params} ({params_str})")
        pow_note = ""
        if HW_GAMMA <= 0.0:
            pow_note = "  [功耗未用代理: HW_GAMMA=0]"
        print(
            f"  HW 代理: latency≈{hw_pred['pred_latency_ms']:.4f} ms (n={hw_pred['latency_norm']:.4f}), "
            f"power≈{hw_pred['pred_power_w']:.4f} W (n={hw_pred['power_norm']:.4f}){pow_note}"
        )
        print(f"{'='*60}")

        if HW_MAX_PRED_LATENCY_MS > 0 and hw_pred["pred_latency_ms"] > HW_MAX_PRED_LATENCY_MS:
            print(
                f"  预测延迟 {hw_pred['pred_latency_ms']:.4f} > 阈值 {HW_MAX_PRED_LATENCY_MS}，跳过训练"
            )
            evaluated_cache[key] = HW_SKIP_LOSS
            evals.append({"params": list(key), "result": HW_SKIP_LOSS})
            _save_hw_cache(cache_path, evals)
            trial_elapsed = time.perf_counter() - trial_start_time
            search_elapsed = time.perf_counter() - search_start_time
            run_elapsed = time.perf_counter() - run_start_time
            _log_row2(
                trial_idx,
                arch_config,
                n_params,
                None,
                None,
                "",
                skipped=True,
                train_info=None,
                hw=hw_pred,
                combined=None,
                hw_skip_reason="pred_latency_threshold",
                trial_elapsed_sec=trial_elapsed,
                time_taken_sec=trial_elapsed,
                search_elapsed_sec=search_elapsed,
                run_elapsed_sec=run_elapsed,
            )
            print(
                f"  用时: trial={trial_elapsed:.3f}s search_total={search_elapsed:.1f}s "
                f"run_total={run_elapsed:.1f}s"
            )
            return HW_SKIP_LOSS

        ks, e, d = (
            arch_config["kernel_size"],
            arch_config["expand_ratio"],
            arch_config["depths"],
        )
        w = arch_config.get("width_multiplier", 1.0)
        ch_part = (
            f"_c{ch['stem_c']}-{ch['stage1_c']}-{ch['stage2_c']}-"
            f"{ch['stage3_c']}-{ch['stage4_c']}-{ch['last_c']}"
        )
        infer_m = arch_config["infer_mode"]
        quant_p = arch_config["quant_precision"]
        infer_part = infer_filename_suffix(infer_m, quant_p)
        d_str = "".join(str(x) for x in d)
        w_str = f"w{w}" if w != 1.0 else "w1"
        checkpoint_path = None
        if models_dir:
            checkpoint_path = os.path.join(
                models_dir,
                f"trial_{trial_idx:03d}_ks{ks}_e{e}_{w_str}_d{d_str}{ch_part}_{params_str}_best{infer_part}.pth",
            )

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

        final_test_acc = (
            history[-1].get("final_test_acc") if history else evaluate(model, test_loader, device)
        )

        if curves_dir and history:
            _plot_trial_curves(history, arch_config, trial_idx, curves_dir)

        final_path = ""
        if models_dir and checkpoint_path:
            final_path = os.path.join(
                models_dir,
                f"trial_{trial_idx:03d}_ks{ks}_e{e}_{w_str}_d{d_str}{ch_part}_{params_str}_test{final_test_acc:.2f}"
                f"{infer_part}.pth",
            )
            if checkpoint_path != final_path:
                os.rename(checkpoint_path, final_path)
            print(f"  模型已保存 → {final_path}")

        comb = _combined_loss(
            float(best_metric),
            hw_pred["latency_norm"],
            hw_pred["power_norm"],
        )
        evaluated_cache[key] = comb
        evals.append({"params": list(key), "result": comb})
        _save_hw_cache(cache_path, evals)
        trial_elapsed = time.perf_counter() - trial_start_time
        search_elapsed = time.perf_counter() - search_start_time
        run_elapsed = time.perf_counter() - run_start_time
        _log_row2(
            trial_idx,
            arch_config,
            n_params,
            final_test_acc,
            best_metric,
            os.path.abspath(final_path) if final_path else "",
            skipped=False,
            train_info=train_info,
            hw=hw_pred,
            combined=comb,
            hw_skip_reason="",
            trial_elapsed_sec=trial_elapsed,
            time_taken_sec=trial_elapsed,
            search_elapsed_sec=search_elapsed,
            run_elapsed_sec=run_elapsed,
        )
        print(
            f"  Trial {trial_idx} 完成: val={best_metric:.4f}  test@best_val={final_test_acc:.4f}  "
            f"combined={comb:.6f} (=α(1-val)+β·lat_n+γ·pow_n)  "
            f"time={trial_elapsed:.3f}s search_total={search_elapsed:.1f}s run_total={run_elapsed:.1f}s\n"
        )
        return comb

    return objective


def run_search_hw():
    run_start_time = time.perf_counter()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    curves_dir = os.path.join(OUTPUT_DIR, "trial_curves") if SEARCH_SAVE_PLOTS else ""
    models_dir = os.path.join(OUTPUT_DIR, "trial_models")
    if curves_dir:
        os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    try:
        proxy = JetsonHWProxy(HW_PROXY_MODEL_DIR)
    except FileNotFoundError as e:
        print(f"[search_hw] 无法加载硬件代理: {e}")
        raise SystemExit(1) from e

    evals = _load_hw_cache(SEARCH_CACHE_PATH)
    x0 = [e["params"] for e in evals]
    y0 = [e["result"] for e in evals]
    bs = SEARCH_BATCH_SIZE if SEARCH_BATCH_SIZE is not None else BATCH_SIZE
    train_loader, val_loader = get_cifar10_loaders(
        root=CIFAR10_ROOT,
        batch_size=bs,
        random_erasing_p=RANDOM_ERASING_P,
    )
    test_loader = get_cifar10_test_loader(root=CIFAR10_ROOT, batch_size=bs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    space = get_search_space()
    objective_fn = _make_objective_hw(
        SEARCH_CALLS,
        curves_dir,
        models_dir,
        evals,
        SEARCH_CACHE_PATH,
        SEARCH_CSV_PATH,
        proxy,
        run_start_time,
        train_loader,
        val_loader,
        test_loader,
        device,
    )

    n_remaining = SEARCH_CALLS - len(evals)
    print(f"\n[search_hw] 硬件反馈搜索（独立目录，不覆盖原版）")
    print(f"  代理目录: {HW_PROXY_MODEL_DIR}")
    print(f"  共 {SEARCH_CALLS} 次评估, 已完成 {len(evals)}, 剩余 {n_remaining}")
    print(f"  α={HW_ALPHA} β={HW_BETA} γ={HW_GAMMA}")
    print(f"  曲线保存: {curves_dir if curves_dir else '关闭'}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  cache: {SEARCH_CACHE_PATH}")
    print(f"  CSV: {SEARCH_CSV_PATH}\n")

    if x0:
        result = gp_minimize(
            objective_fn,
            space,
            n_calls=SEARCH_CALLS,
            random_state=42,
            verbose=True,
            x0=x0,
            y0=y0,
            n_initial_points=-len(x0),
        )
    else:
        result = gp_minimize(
            objective_fn,
            space,
            n_calls=SEARCH_CALLS,
            random_state=42,
            verbose=True,
        )

    best_params = result.x
    best_config = params_to_arch_config(best_params)
    best_combined = result.fun

    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n[search_hw] 最优配置 → {BEST_CONFIG_PATH}")
    print(f"  最小 combined_loss: {best_combined:.6f}")
    print(f"  架构: {best_config}")
    return best_config
