"""
Train NASYOLOv8 detector on BDD100K (or COCO128 for quick test).

Uses Ultralytics dataset/loss utilities with a standard PyTorch training loop.

Usage:
    conda activate nas_det

    # Modify the Config section below, then run directly:
    python train_nas_yolo.py

    # Or override any param via command line:
    python train_nas_yolo.py --quick
    python train_nas_yolo.py --backbone ../results/best_model_test0.90.pth --epochs 100 --batch 8 --device 1
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ── CUDA acceleration ────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True       # auto-tune conv algorithms
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 on Ampere+ GPUs
torch.backends.cudnn.allow_tf32 = True

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import unwrap_model as de_parallel

from torchvision.ops import nms as torchvision_nms
from backbones import (
    BACKBONE_CHOICES,
    MOBILENETV3_IMAGENET,
    NAS_BACKBONE,
    describe_backbone,
)
from experiment_report import (
    append_metrics_row,
    benchmark_forward_fps,
    count_params_m,
    count_trainable_params_m,
    upsert_summary_row,
)
from nas_yolo import NASYOLOv8

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    _HAS_TORCHMETRICS = True
except (ImportError, ModuleNotFoundError):
    _HAS_TORCHMETRICS = False
    print("[WARN] mAP disabled. Install with: pip install torchmetrics[detection]")

_HERE = Path(__file__).resolve().parent

# ── Config (modify here or override via command line) ─────────────────────
BACKBONE_TYPE = NAS_BACKBONE  #"mobilenetv3_imagenet", "nas"
BACKBONE     = None
NAS_BACKBONE_CKPT = "../results/trial_017_ks3_e7_w2.0_d4422_2.4M_test0.91_infer_dla3gpu_fp16.pth"
PRETRAINED_DIR = "pretrained"
DATA_YAML    = str(_HERE / "bdd100k.yaml")
EPOCHS       = 300
BATCH        = 16          # reduce to 8 if OOM
IMGSZ        = 640
WORKERS      = 8
DEVICE       = [0]   # single GPU: 0 | multi-GPU: [0,1,2] | CPU: "cpu"
LR           = 0.01
MOMENTUM     = 0.937
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS= 3
UNFREEZE_EPOCH = 5         # epochs to keep backbone frozen
AMP            = True      # mixed precision training (fp16) — ~1.5x speedup
ACCUM_STEPS    = 1         # gradient accumulation steps  默认是 1，也就是不开启梯度累积
SAVE_PERIOD  = 5          # save checkpoint every N epochs
VAL_PERIOD     = 5         # run val loss every N epochs (1 = every epoch)
VAL_MAP_PERIOD = 10         # compute mAP every N epochs (0 = disable)
BEST_CKPT_METRIC    = "map5095"  # val_loss | map50 | map5095
EARLY_STOP_METRIC   = "val_loss"  # val_loss | map50 | map5095
EARLY_STOP_PATIENCE = 20         # monitored metric checks without improvement
EARLY_STOP_MIN_DELTA = 1e-4
PROJECT      = "runs/bdd100k_nas_yolo-0.91"
RUN_NAME     = "nas_yolo_baseline"
P3_CH        = 96
P4_CH        = 192
P5_CH        = 192
# ── Quick-test overrides (used with --quick) ──────────────────────────────
QUICK_DATA    = "coco128.yaml"
QUICK_EPOCHS  = 100
QUICK_PROJECT = "runs/coco128_nas_yolo"
QUICK_NAME    = "nas_yolo_test"
COMPARE_CSV   = str(_HERE / "results_compare.csv")
FPS_BENCHMARK_BATCHES = 30
FPS_WARMUP_BATCHES = 5
# ─────────────────────────────────────────────────────────────────────────


def default_run_name(backbone_type: str, quick: bool = False) -> str:
    suffix = "_test" if quick else ""
    return f"nas_yolo_{backbone_type}{suffix}"


def build_model(
    backbone_type: str,
    backbone_ckpt: str | None,
    nc: int,
    device,
    backbone_pretrained: bool,
    pretrained_dir: str,
    p3_ch: int,
    p4_ch: int,
    p5_ch: int,
) -> NASYOLOv8:
    model = NASYOLOv8.from_backbone(
        backbone_type=backbone_type,
        backbone_ckpt=backbone_ckpt,
        nc=nc,
        backbone_pretrained=backbone_pretrained,
        pretrained_dir=pretrained_dir,
        p3_ch=p3_ch,
        p4_ch=p4_ch,
        p5_ch=p5_ch,
    )
    model = model.to(device)

    # ── Detailed parameter summary ────────────────────────────────────────
    def _count(module):
        return sum(p.numel() for p in module.parameters()) / 1e6

    n_backbone = _count(model.backbone)
    n_proj     = sum(_count(m) for m in [model.proj_c3, model.proj_c4, model.proj_c5])
    n_sppf     = _count(model.sppf)
    n_neck     = sum(_count(m) for m in [model.c2f_p4, model.c2f_p3,
                                          model.down1, model.c2f_n4,
                                          model.down2, model.c2f_n5])
    n_head     = _count(model.detect)
    n_total    = sum(p.numel() for p in model.parameters()) / 1e6

    c3, c4, c5 = model.backbone.out_channels
    backbone_desc = describe_backbone(getattr(model, "_backbone_spec", None))
    print(f"\n{'='*50}")
    print(f"  NASYOLOv8 Model Summary")
    print(f"{'='*50}")
    print(f"  Backbone : {n_backbone:.4f}M  ({backbone_desc})")
    print(f"    C3={c3}ch  C4={c4}ch  C5={c5}ch")
    print(f"  Proj     : {n_proj:.4f}M  (to P3={p3_ch}, P4={p4_ch}, P5={p5_ch})")
    print(f"  SPPF     : {n_sppf:.4f}M")
    print(f"  Neck     : {n_neck:.4f}M  (PAN-FPN with C2f)")
    print(f"  Head     : {n_head:.4f}M  (Detect, nc={nc})")
    print(f"  ─────────────────────────────")
    print(f"  Total    : {n_total:.4f}M")
    print(f"{'='*50}\n")
    return model


def get_loaders(data_yaml: str, imgsz: int, batch: int, workers: int):
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.data.utils import check_det_dataset

    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data_yaml; cfg.imgsz = imgsz
    cfg.batch = batch;    cfg.workers = workers
    cfg.task  = "detect"

    data_info    = check_det_dataset(data_yaml)
    nc           = data_info["nc"]
    train_ds     = build_yolo_dataset(cfg, data_info["train"], batch, data_info, mode="train", rect=False)
    val_ds       = build_yolo_dataset(cfg, data_info["val"],   batch, data_info, mode="val",   rect=False)
    train_loader = build_dataloader(train_ds, batch, workers, shuffle=True,  rank=-1)
    val_loader   = build_dataloader(val_ds,   batch, workers, shuffle=False, rank=-1)
    return train_loader, val_loader, nc


def compute_map(model, val_loader, device, conf=0.25, iou=0.45):
    """Compute mAP50 and mAP50-95 on the validation set."""
    if not _HAS_TORCHMETRICS:
        return None, None
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="  mAP eval", leave=False, ncols=80):
            imgs = batch_data["img"].to(device).float() / 255.0
            batch_idx  = batch_data["batch_idx"].to(device)
            gt_bboxes  = batch_data["bboxes"].to(device)   # (N, 4) xywhn
            gt_cls     = batch_data["cls"].to(device).long().squeeze(-1)
            bs         = imgs.shape[0]

            preds = model(imgs)
            pred_tensor = preds[0] if isinstance(preds, (list, tuple)) else preds

            preds_list  = []
            targets_list= []

            for i in range(bs):
                # ── Predictions ───────────────────────────────────────────
                p = pred_tensor[i]                     # (nc+4, 8400)
                boxes_xywh = p[:4]
                cls_scores  = p[4:]
                x1 = (boxes_xywh[0] - boxes_xywh[2] / 2)
                y1 = (boxes_xywh[1] - boxes_xywh[3] / 2)
                x2 = (boxes_xywh[0] + boxes_xywh[2] / 2)
                y2 = (boxes_xywh[1] + boxes_xywh[3] / 2)
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
                h, w = imgs.shape[2], imgs.shape[3]
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clamp(0, w)
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clamp(0, h)
                scores, labels = cls_scores.max(dim=0)
                mask = scores > conf
                boxes_xyxy = boxes_xyxy[mask]
                scores     = scores[mask]
                labels     = labels[mask]
                if len(scores):
                    # Use class-aware NMS to avoid suppressing overlapping boxes
                    # from different categories, which would under-estimate mAP.
                    max_wh = max(h, w) + 1
                    nms_boxes = boxes_xyxy + labels.unsqueeze(1).to(boxes_xyxy.dtype) * max_wh
                    keep = torchvision_nms(nms_boxes, scores, iou)
                    boxes_xyxy = boxes_xyxy[keep]
                    scores     = scores[keep]
                    labels     = labels[keep]
                preds_list.append({"boxes": boxes_xyxy, "scores": scores, "labels": labels})

                # ── Ground truth ──────────────────────────────────────────
                gt_mask = batch_idx == i
                gb = gt_bboxes[gt_mask]              # (M, 4) xywhn
                gc = gt_cls[gt_mask]
                # xywhn → xyxy (pixel)
                gx1 = (gb[:, 0] - gb[:, 2] / 2) * w
                gy1 = (gb[:, 1] - gb[:, 3] / 2) * h
                gx2 = (gb[:, 0] + gb[:, 2] / 2) * w
                gy2 = (gb[:, 1] + gb[:, 3] / 2) * h
                gt_boxes_xyxy = torch.stack([gx1, gy1, gx2, gy2], dim=1)
                targets_list.append({"boxes": gt_boxes_xyxy, "labels": gc})

            metric.update(preds_list, targets_list)

    results = metric.compute()
    map50   = results["map_50"].item()
    map5095 = results["map"].item()
    return map50, map5095


def _monitor_mode(metric_name: str) -> str:
    if metric_name == "val_loss":
        return "min"
    if metric_name in {"map50", "map5095"}:
        return "max"
    raise ValueError(f"Unsupported early stop metric: {metric_name}")


def _monitor_improved(metric_name: str, current: float, best: float, min_delta: float) -> bool:
    if _monitor_mode(metric_name) == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def _csv_value(value):
    return "" if value is None else value


def train(backbone_type, backbone_ckpt, data_yaml, epochs, project, run_name,
          device_id, batch, imgsz, workers, lr, momentum, weight_decay,
          warmup_epochs, unfreeze_epoch, save_period, backbone_pretrained,
          pretrained_dir, p3_ch, p4_ch, p5_ch, resume_ckpt=None,
          accum_steps=ACCUM_STEPS,
          best_ckpt_metric=BEST_CKPT_METRIC,
          early_stop_metric=EARLY_STOP_METRIC,
          early_stop_patience=EARLY_STOP_PATIENCE,
          early_stop_min_delta=EARLY_STOP_MIN_DELTA,
          compare_csv=COMPARE_CSV,
          fps_benchmark_batches=FPS_BENCHMARK_BATCHES,
          fps_warmup_batches=FPS_WARMUP_BATCHES):

    # ── Device setup ─────────────────────────────────────────────────────
    # Normalize command-line string input (e.g. "0", "0,1", "cpu")
    if isinstance(device_id, str):
        device_id = device_id.strip()
        if device_id.lower() == "cpu":
            device_id = "cpu"
        elif "," in device_id:
            try:
                device_id = [int(x.strip()) for x in device_id.split(",")]
            except ValueError:
                device_id = "cpu"
        elif device_id.isdigit():
            device_id = int(device_id)
        else:
            device_id = "cpu"

    if isinstance(device_id, list):
        device     = torch.device(f"cuda:{device_id[0]}")
        multi_gpu  = True
        gpu_ids    = device_id
    elif isinstance(device_id, int):
        device     = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        multi_gpu  = False
        gpu_ids    = [device_id]
    else:
        device     = torch.device("cpu")
        multi_gpu  = False
        gpu_ids    = []

    save_dir = (Path(project) / run_name).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = save_dir / "metrics.csv"
    print(f"Device: {gpu_ids if multi_gpu else device}  |  Saving to: {save_dir}")
    accum_steps = max(1, int(accum_steps))

    if early_stop_metric.startswith("map") and not _HAS_TORCHMETRICS:
        raise RuntimeError("mAP early stopping requires torchmetrics[detection] to be installed")
    if early_stop_metric.startswith("map") and VAL_MAP_PERIOD <= 0:
        raise ValueError("mAP early stopping requires VAL_MAP_PERIOD > 0")
    if best_ckpt_metric.startswith("map") and not _HAS_TORCHMETRICS:
        print("[WARN] mAP best-checkpoint selection disabled because torchmetrics is unavailable; falling back to val_loss")
        best_ckpt_metric = "val_loss"
    if best_ckpt_metric.startswith("map") and VAL_MAP_PERIOD <= 0:
        print("[WARN] mAP best-checkpoint selection disabled because VAL_MAP_PERIOD <= 0; falling back to val_loss")
        best_ckpt_metric = "val_loss"

    train_loader, val_loader, nc = get_loaders(data_yaml, imgsz, batch, workers)
    print(f"Dataset: {len(train_loader.dataset)} train  {len(val_loader.dataset)} val  nc={nc}")

    model     = build_model(
        backbone_type,
        backbone_ckpt,
        nc,
        device,
        backbone_pretrained,
        pretrained_dir,
        p3_ch,
        p4_ch,
        p5_ch,
    )
    criterion = v8DetectionLoss(de_parallel(model))

    # Multi-GPU with DataParallel
    if multi_gpu:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"DataParallel: {len(gpu_ids)} GPUs {gpu_ids}")

    # AMP (mixed precision)
    use_amp = AMP and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)
    print(f"AMP: {'ON (fp16)' if use_amp else 'OFF'}")

    raw_model = de_parallel(model)   # unwrapped model for saving/freezing
    model_params_m = count_params_m(raw_model)
    model_trainable_params_m = count_trainable_params_m(raw_model)
    backbone_desc = describe_backbone(getattr(raw_model, "_backbone_spec", None))

    # ── Resume: load training state ───────────────────────────────────────
    start_epoch   = 1
    best_val_loss = float("inf")
    best_ckpt_value = float("inf") if _monitor_mode(best_ckpt_metric) == "min" else float("-inf")
    best_monitor_value = float("inf") if _monitor_mode(early_stop_metric) == "min" else float("-inf")
    best_map50 = None
    best_map5095 = None
    last_map50 = None
    last_map5095 = None
    early_stop_wait = 0
    saved_state   = None
    if resume_ckpt is not None:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        print(f"Resuming from {ckpt_path}")
        saved_state = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(saved_state["model_state"])
        start_epoch   = saved_state["epoch"] + 1
        best_val_loss = saved_state["best_val_loss"]
        best_ckpt_value = saved_state.get(
            "best_ckpt_value",
            best_val_loss if best_ckpt_metric == "val_loss" else best_ckpt_value,
        )
        best_monitor_value = saved_state.get(
            "best_monitor_value",
            best_val_loss if early_stop_metric == "val_loss" else best_monitor_value,
        )
        best_map50 = saved_state.get("best_map50")
        best_map5095 = saved_state.get("best_map5095")
        last_map50 = saved_state.get("last_map50")
        last_map5095 = saved_state.get("last_map5095")
        early_stop_wait = saved_state.get("early_stop_wait", 0)
        print(f"  → Resuming at epoch {start_epoch}/{epochs}, best_val_loss={best_val_loss:.4f}")

    # ── Backbone freeze / unfreeze setup ──────────────────────────────────
    # If already past unfreeze_epoch in a resumed run, keep backbone unfrozen
    # Unfreeze happens at epoch == unfreeze_epoch + 1, so a checkpoint saved at
    # epoch == unfreeze_epoch still has the backbone frozen and a single optimizer group.
    backbone_unfrozen = saved_state is not None and saved_state["epoch"] > unfreeze_epoch

    # Always freeze backbone first, build optimizer from non-backbone params only
    for p in raw_model.backbone.parameters():
        p.requires_grad = False

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    if backbone_unfrozen:
        # Replay the same unfreeze flow so param groups match the checkpoint exactly
        for p in raw_model.backbone.parameters():
            p.requires_grad = True
        optimizer.add_param_group({"params": list(raw_model.backbone.parameters()),
                                   "lr": lr * 0.1})
        backbone_unfrozen = True

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    if saved_state is not None:
        optimizer.load_state_dict(saved_state["optimizer_state"])
        scheduler.load_state_dict(saved_state["scheduler_state"])
        scaler.load_state_dict(saved_state["scaler_state"])
        print(f"  → Optimizer / scheduler / scaler states restored")

    val_map_period = VAL_MAP_PERIOD
    print(f"Best checkpoint: metric={best_ckpt_metric}")
    print(f"Early stop: metric={early_stop_metric}  patience={early_stop_patience}  min_delta={early_stop_min_delta}")
    if accum_steps > 1:
        print(f"Gradient accumulation: {accum_steps} steps  (effective batch ~= {batch * accum_steps})")
    header = (f"{'Epoch':>8}  {'box':>8}  {'cls':>8}  {'dfl':>8}  {'train':>8}"
              f"  {'val_box':>8}  {'val_cls':>8}  {'val':>8}"
              f"  {'mAP50':>8}  {'mAP5095':>8}  {'lr':>10}")
    print(header)
    print("-" * len(header))

    stopped_early = False
    stop_reason = ""
    for epoch in range(start_epoch, epochs + 1):

        if not backbone_unfrozen and epoch == unfreeze_epoch + 1:
            for p in raw_model.backbone.parameters():
                p.requires_grad = True
            optimizer.add_param_group({"params": list(raw_model.backbone.parameters()),
                                       "lr": lr * 0.1})
            backbone_unfrozen = True
            print(f"  → Backbone unfrozen (backbone lr={lr*0.1:.5f})")

        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * epoch / warmup_epochs

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        sum_loss  = torch.zeros(3, device=device)   # box, cls, dfl
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [train]",
                    leave=False, ncols=80)
        num_batches = len(train_loader)
        for batch_i, batch_data in enumerate(pbar, start=1):
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch_data.items()}
            with autocast(enabled=use_amp):
                preds = model(batch_data["img"].float() / 255.0)
                loss, loss_items = criterion(preds, batch_data)
            loss_total = loss.sum()
            scaler.scale(loss_total / accum_steps).backward()
            should_step = (batch_i % accum_steps == 0) or (batch_i == num_batches)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            sum_loss += loss_items.detach()
            pbar.set_postfix(loss=f"{loss_total.item():.3f}")

        avg_train = sum_loss / len(train_loader)   # [box, cls, dfl]

        # ── Validate (every VAL_PERIOD epochs) ─────────────────────────────
        run_val = (epoch % VAL_PERIOD == 0) or (epoch == epochs)
        avg_val = torch.zeros(3, device=device)
        val_total = 0.0

        if run_val:
            model.eval()
            val_criterion = v8DetectionLoss(raw_model)
            sum_val = torch.zeros(3, device=device)
            with torch.no_grad():
                for batch_data in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [val]  ",
                                       leave=False, ncols=80):
                    batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                  for k, v in batch_data.items()}
                    with autocast(enabled=use_amp):
                        preds = model(batch_data["img"].float() / 255.0)
                        _, loss_items = val_criterion(preds, batch_data)
                    sum_val += loss_items.detach()
            avg_val = sum_val / len(val_loader)
            val_total = avg_val.sum().item()

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        # mAP (every val_map_period epochs)
        map50, map5095 = (None, None)
        if val_map_period > 0 and ((epoch % val_map_period == 0) or (epoch == epochs)):
            map50, map5095 = compute_map(model, val_loader, device)

        map50_str   = f"{map50:>8.4f}"   if map50   is not None else f"{'--':>8}"
        map5095_str = f"{map5095:>8.4f}" if map5095 is not None else f"{'--':>8}"
        val_box_str = f"{avg_val[0].item():>8.4f}" if run_val else f"{'--':>8}"
        val_cls_str = f"{avg_val[1].item():>8.4f}" if run_val else f"{'--':>8}"
        val_str     = f"{val_total:>8.4f}"         if run_val else f"{'--':>8}"

        if map50 is not None:
            last_map50 = map50
            best_map50 = map50 if best_map50 is None else max(best_map50, map50)
        if map5095 is not None:
            last_map5095 = map5095
            best_map5095 = map5095 if best_map5095 is None else max(best_map5095, map5095)

        print(f"{epoch:>8d}"
              f"  {avg_train[0].item():>8.4f}"
              f"  {avg_train[1].item():>8.4f}"
              f"  {avg_train[2].item():>8.4f}"
              f"  {avg_train.sum().item():>8.4f}"
              f"  {val_box_str}"
              f"  {val_cls_str}"
              f"  {val_str}"
              f"  {map50_str}"
              f"  {map5095_str}"
              f"  {cur_lr:>10.6f}", flush=True)

        append_metrics_row(
            metrics_csv_path,
            {
                "epoch": epoch,
                "train_box": avg_train[0].item(),
                "train_cls": avg_train[1].item(),
                "train_dfl": avg_train[2].item(),
                "train_total": avg_train.sum().item(),
                "val_box": _csv_value(avg_val[0].item() if run_val else None),
                "val_cls": _csv_value(avg_val[1].item() if run_val else None),
                "val_total": _csv_value(val_total if run_val else None),
                "map50": _csv_value(map50),
                "map5095": _csv_value(map5095),
                "lr": cur_lr,
            },
        )

        if run_val and val_total < best_val_loss:
            best_val_loss = val_total

        ckpt_metric_value = None
        if best_ckpt_metric == "val_loss" and run_val:
            ckpt_metric_value = val_total
        elif best_ckpt_metric == "map50" and map50 is not None:
            ckpt_metric_value = map50
        elif best_ckpt_metric == "map5095" and map5095 is not None:
            ckpt_metric_value = map5095

        if ckpt_metric_value is not None and _monitor_improved(
            best_ckpt_metric, ckpt_metric_value, best_ckpt_value, early_stop_min_delta
        ):
            best_ckpt_value = ckpt_metric_value
            raw_model.save(str(save_dir / "best.pt"), backbone_ckpt=backbone_ckpt)
            print(f"  ↑ best {best_ckpt_metric} → {save_dir}/best.pt")

        monitor_value = None
        if early_stop_metric == "val_loss" and run_val:
            monitor_value = val_total
        elif early_stop_metric == "map50" and map50 is not None:
            monitor_value = map50
        elif early_stop_metric == "map5095" and map5095 is not None:
            monitor_value = map5095

        if monitor_value is not None:
            if _monitor_improved(early_stop_metric, monitor_value, best_monitor_value, early_stop_min_delta):
                best_monitor_value = monitor_value
                early_stop_wait = 0
            else:
                early_stop_wait += 1
                print(f"  · no {early_stop_metric} improvement for {early_stop_wait}/{early_stop_patience} checks")

                if early_stop_patience > 0 and early_stop_wait >= early_stop_patience:
                    stopped_early = True
                    stop_reason = (f"Early stopping triggered at epoch {epoch}: "
                                   f"{early_stop_metric} did not improve for {early_stop_patience} checks")

        # ── Periodic checkpoint (model weights + full training state) ─────
        if epoch % save_period == 0 or epoch == epochs:
            ckpt_path = str(save_dir / f"epoch{epoch}.pt")
            torch.save({
                "epoch":           epoch,
                "model_state":     raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state":    scaler.state_dict(),
                "best_val_loss":   best_val_loss,
                "best_ckpt_metric": best_ckpt_metric,
                "best_ckpt_value": best_ckpt_value,
                "best_monitor_value": best_monitor_value,
                "best_map50": best_map50,
                "best_map5095": best_map5095,
                "last_map50": last_map50,
                "last_map5095": last_map5095,
                "early_stop_wait": early_stop_wait,
                "early_stop_metric": early_stop_metric,
                "backbone_spec":   getattr(raw_model, "_backbone_spec", None),
                "nc":              raw_model.nc,
            }, ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

        if stopped_early:
            print(f"  → {stop_reason}")
            break

    raw_model.save(str(save_dir / "last.pt"), backbone_ckpt=backbone_ckpt)
    raw_model.eval()
    fps, latency_ms, fps_batches, fps_images = benchmark_forward_fps(
        loader=val_loader,
        device=device,
        prepare_batch_fn=lambda batch_data: {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_data.items()
        },
        forward_fn=lambda batch_data: raw_model(batch_data["img"].float() / 255.0),
        max_batches=fps_benchmark_batches,
        warmup_batches=fps_warmup_batches,
    )
    upsert_summary_row(
        compare_csv,
        {
            "task": "detect",
            "detector": "yolo",
            "backbone_type": backbone_type,
            "backbone_desc": backbone_desc,
            "data": data_yaml,
            "project": project,
            "run_name": run_name,
            "run_dir": str(save_dir),
            "epochs_target": epochs,
            "epochs_completed": epoch,
            "batch": batch,
            "imgsz": imgsz,
            "device": str(gpu_ids if multi_gpu else device),
            "amp": use_amp,
            "pretrained_backbone": backbone_pretrained,
            "params_m": model_params_m,
            "trainable_params_m": model_trainable_params_m,
            "best_val_loss": best_val_loss,
            "best_ckpt_metric": best_ckpt_metric,
            "best_ckpt_value": best_ckpt_value,
            "best_map50": _csv_value(best_map50),
            "best_map5095": _csv_value(best_map5095),
            "last_map50": _csv_value(last_map50),
            "last_map5095": _csv_value(last_map5095),
            "fps": _csv_value(fps),
            "latency_ms_per_image": _csv_value(latency_ms),
            "fps_batches": fps_batches,
            "fps_images": fps_images,
            "resume_ckpt": _csv_value(resume_ckpt),
            "best_ckpt_path": str(save_dir / "best.pt"),
            "last_ckpt_path": str(save_dir / "last.pt"),
            "metrics_csv": str(metrics_csv_path),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        },
    )
    print(
        f"\nDone. Best val loss: {best_val_loss:.4f}"
        f"  |  Best {best_ckpt_metric}: {best_ckpt_value:.4f}"
        f"  |  Early stop best {early_stop_metric}: {best_monitor_value:.4f}"
        f"  |  Weights: {save_dir}"
        f"  |  Compare CSV: {compare_csv}"
    )


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--backbone",       default=BACKBONE)
    pa.add_argument("--backbone-type",  default=BACKBONE_TYPE, choices=BACKBONE_CHOICES)
    pa.add_argument("--pretrained-dir", default=PRETRAINED_DIR,
                    help="Directory under current workspace to store downloaded torchvision weights")
    pa.add_argument("--no-pretrained",  action="store_true",
                    help="Disable ImageNet pretrained weights for torchvision backbones")
    pa.add_argument("--data",           default=DATA_YAML)
    pa.add_argument("--epochs",         type=int,   default=EPOCHS)
    pa.add_argument("--batch",          type=int,   default=BATCH)
    pa.add_argument("--imgsz",          type=int,   default=IMGSZ)
    pa.add_argument("--workers",        type=int,   default=WORKERS)
    pa.add_argument("--device",         default=DEVICE)
    pa.add_argument("--lr",             type=float, default=LR)
    pa.add_argument("--accum-steps",    type=int, default=ACCUM_STEPS,
                    help="gradient accumulation steps; effective batch ~= batch * accum_steps")
    pa.add_argument("--project",        default=PROJECT)
    pa.add_argument("--name",           default=RUN_NAME)
    pa.add_argument("--best-ckpt-metric", default=BEST_CKPT_METRIC,
                    choices=["val_loss", "map50", "map5095"])
    pa.add_argument("--early-stop-metric", default=EARLY_STOP_METRIC,
                    choices=["val_loss", "map50", "map5095"])
    pa.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE,
                    help="Number of monitored metric checks without improvement before stopping")
    pa.add_argument("--early-stop-min-delta", type=float, default=EARLY_STOP_MIN_DELTA,
                    help="Minimum improvement required to reset early stopping patience")
    pa.add_argument("--quick",          action="store_true", help="Quick test with COCO128")
    pa.add_argument("--p3-ch",          type=int, default=P3_CH)
    pa.add_argument("--p4-ch",          type=int, default=P4_CH)
    pa.add_argument("--p5-ch",          type=int, default=P5_CH)
    pa.add_argument("--resume",         type=str, default=None,
                    help="Optional checkpoint path to continue training from")
    pa.add_argument("--compare-csv",    default=COMPARE_CSV,
                    help="Path to aggregated experiment summary CSV")
    pa.add_argument("--fps-batches",    type=int, default=FPS_BENCHMARK_BATCHES,
                    help="Validation batches to benchmark inference FPS after training (0 disables)")
    pa.add_argument("--fps-warmup-batches", type=int, default=FPS_WARMUP_BATCHES,
                    help="Warmup batches before FPS timing")
    args = pa.parse_args()

    if args.quick:
        args.data    = QUICK_DATA
        args.epochs  = QUICK_EPOCHS
        args.project = QUICK_PROJECT
        if args.name == RUN_NAME:
            args.name = default_run_name(args.backbone_type, quick=True)
    elif args.name == RUN_NAME:
        args.name = default_run_name(args.backbone_type, quick=False)

    if args.backbone_type == NAS_BACKBONE and not args.backbone:
        args.backbone = NAS_BACKBONE_CKPT

    train(
        backbone_type  = args.backbone_type,
        backbone_ckpt  = args.backbone,
        data_yaml      = args.data,
        epochs         = args.epochs,
        project        = args.project,
        run_name       = args.name,
        device_id      = args.device,
        batch          = args.batch,
        imgsz          = args.imgsz,
        workers        = args.workers,
        lr             = args.lr,
        momentum       = MOMENTUM,
        weight_decay   = WEIGHT_DECAY,
        warmup_epochs  = WARMUP_EPOCHS,
        unfreeze_epoch = UNFREEZE_EPOCH,
        save_period    = SAVE_PERIOD,
        backbone_pretrained = not args.no_pretrained,
        pretrained_dir = args.pretrained_dir,
        p3_ch          = args.p3_ch,
        p4_ch          = args.p4_ch,
        p5_ch          = args.p5_ch,
        resume_ckpt    = args.resume,
        accum_steps    = args.accum_steps,
        best_ckpt_metric = args.best_ckpt_metric,
        early_stop_metric = args.early_stop_metric,
        early_stop_patience = args.early_stop_patience,
        early_stop_min_delta = args.early_stop_min_delta,
        compare_csv    = args.compare_csv,
        fps_benchmark_batches = args.fps_batches,
        fps_warmup_batches = args.fps_warmup_batches,
    )
