#!/usr/bin/env python3
"""
第一阶段检测训练启动脚本
统一配置 DAIR-V2X 数据集，支持 NAS backbone 和 MobileNetV3 baseline。

Usage:
    # NAS backbone
    python run_phase1_training.py --mode nas --epochs 300 --device 0

    # MobileNetV3-Large baseline
    python run_phase1_training.py --mode baseline --epochs 300 --device 0
"""
import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent

# ------------------------------------------------------------------
# 统一配置
# ------------------------------------------------------------------
DATA_YAML = str(_PROJECT_ROOT / "datasets" / "dair_v2x_yolo" / "dair_v2x.yaml")
NAS_BACKBONE_CKPT = str(_PROJECT_ROOT / "results" / "trial_017_ks3_e7_w2.0_d4422_2.4M_test0.91_infer_dla3gpu_fp16.pth")

# 训练超参（对两者保持一致，确保公平对比）
EPOCHS = 300
BATCH = 16
IMGSZ = 640
WORKERS = 8
LR = 0.01
MOMENTUM = 0.937
WEIGHT_DECAY = 5e-4
WARMUP_EPOCHS = 3
UNFREEZE_EPOCH = 5
AMP = True
SAVE_PERIOD = 5
VAL_PERIOD = 5
VAL_MAP_PERIOD = 10
BEST_CKPT_METRIC = "map5095"
EARLY_STOP_METRIC = "map5095"
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 1e-4
P3_CH = 128
P4_CH = 256
P5_CH = 256


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--mode", required=True, choices=["nas", "baseline"],
                    help="训练模式: nas=使用NAS backbone, baseline=MobileNetV3-Large")
    pa.add_argument("--epochs", type=int, default=EPOCHS)
    pa.add_argument("--batch", type=int, default=BATCH)
    pa.add_argument("--imgsz", type=int, default=IMGSZ)
    pa.add_argument("--device", default="0")
    pa.add_argument("--workers", type=int, default=WORKERS)
    pa.add_argument("--project", default="runs/dair_v2x")
    pa.add_argument("--name", default=None,
                    help="运行名称；默认 nas->nas_yolo, baseline->mobilenetv3_yolo")
    pa.add_argument("--resume", default=None,
                    help="从指定 checkpoint 断点续训")
    pa.add_argument("--accum-steps", type=int, default=1,
                    help="梯度累积步数，effective batch = batch * accum_steps")
    pa.add_argument("--dry-run", action="store_true",
                    help="只打印配置，不启动训练")
    args = pa.parse_args()

    if args.mode == "nas":
        backbone_type = "nas"
        backbone_ckpt = NAS_BACKBONE_CKPT
        default_name = "nas_yolo"
    else:
        backbone_type = "mobilenetv3_imagenet"
        backbone_ckpt = None
        default_name = "mobilenetv3_yolo"

    run_name = args.name or default_name

    # 构建 train_nas_yolo.py 的命令行参数
    cmd = [
        sys.executable, str(_HERE / "train_nas_yolo.py"),
        "--backbone-type", backbone_type,
        "--data", DATA_YAML,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--imgsz", str(args.imgsz),
        "--device", str(args.device),
        "--workers", str(args.workers),
        "--lr", str(LR),
        "--project", args.project,
        "--name", run_name,
        "--best-ckpt-metric", BEST_CKPT_METRIC,
        "--early-stop-metric", EARLY_STOP_METRIC,
        "--early-stop-patience", str(EARLY_STOP_PATIENCE),
        "--early-stop-min-delta", str(EARLY_STOP_MIN_DELTA),
        "--p3-ch", str(P3_CH),
        "--p4-ch", str(P4_CH),
        "--p5-ch", str(P5_CH),
    ]

    if backbone_ckpt:
        cmd += ["--backbone", backbone_ckpt]
    if args.resume:
        cmd += ["--resume", args.resume]
    if args.accum_steps > 1:
        cmd += ["--accum-steps", str(args.accum_steps)]

    if not args.dry_run:
        import subprocess
        print("=" * 60)
        print(f"  Launching Phase-1 training: {args.mode}")
        print(f"  Backbone: {backbone_type}")
        print(f"  Data: {DATA_YAML}")
        print(f"  Project: {args.project}/{run_name}")
        accum_info = f"  Accum: {args.accum_steps} (effective batch={args.batch * args.accum_steps})" if args.accum_steps > 1 else ""
        print(f"  Epochs: {args.epochs}  Batch: {args.batch}  imgsz: {args.imgsz}{accum_info}")
        print(f"  Device: {args.device}")
        if args.resume:
            print(f"  Resume: {args.resume}")
        print("=" * 60)
        subprocess.run(cmd, check=True)
    else:
        print("[DRY RUN] Command:")
        print(" ".join(cmd))


if __name__ == "__main__":
    main()
