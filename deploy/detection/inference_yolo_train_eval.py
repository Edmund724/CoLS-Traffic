"""
Evaluate NAS YOLO with the exact same dataloader and mAP decode logic used in train_nas_yolo.py.

Usage:
    python inference_yolo_train_eval.py --split val
    python inference_yolo_train_eval.py --split both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG

from nas_yolo import NASYOLOv8
from train_nas_yolo import compute_map

_HERE = Path(__file__).resolve().parent

WEIGHTS = "runs/bdd100k_nas_yolo-0.91/nas_yolo_nas/best.pt"
DATA_YAML = str(_HERE / "bdd100k.yaml")
IMGSZ = 640
BATCH_EVAL = 16
WORKERS = 8
SPLIT = "val"


def get_eval_loader(data_yaml: str, imgsz: int, batch: int, workers: int, split: str):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data_yaml
    cfg.imgsz = imgsz
    cfg.batch = batch
    cfg.workers = workers
    cfg.task = "detect"

    data_info = check_det_dataset(data_yaml)
    nc = data_info["nc"]
    if split == "val":
        img_path = data_info["val"]
    else:
        if "test" not in data_info:
            raise KeyError(
                "Split 'test' is not defined in the dataset yaml. "
                "Add a 'test:' entry to bdd100k.yaml or run with --split val."
            )
        img_path = data_info["test"]

    eval_ds = build_yolo_dataset(cfg, img_path, batch, data_info, mode="val", rect=True)
    eval_loader = build_dataloader(eval_ds, batch, workers, shuffle=False, rank=-1)
    return eval_loader, nc


def evaluate(weights: str, data_yaml: str, split: str, imgsz: int, batch_eval: int, workers: int):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NASYOLOv8.load(weights).to(device).eval()

    eval_loader, nc = get_eval_loader(data_yaml, imgsz, batch_eval, workers, split)
    print(f"\nEvaluating split={split}  images={len(eval_loader.dataset)}  imgsz={imgsz}  batch={batch_eval}")

    map50, map5095 = compute_map(model, eval_loader, device)

    print(f"\n{'=' * 65}")
    print(f"  Exact Train-Style Evaluation  ({split})")
    print(f"{'=' * 65}")
    print(f"  mAP@0.50      : {map50:.4f}")
    print(f"  mAP@0.50:0.95 : {map5095:.4f}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--weights", default=WEIGHTS)
    pa.add_argument("--data", default=DATA_YAML)
    pa.add_argument("--split", default=SPLIT, choices=["val", "test", "both"])
    pa.add_argument("--imgsz", type=int, default=IMGSZ)
    pa.add_argument("--batch_eval", type=int, default=BATCH_EVAL)
    pa.add_argument("--workers", type=int, default=WORKERS)
    args = pa.parse_args()

    splits = ["val", "test"] if args.split == "both" else [args.split]
    for split_name in splits:
        evaluate(
            weights=args.weights,
            data_yaml=args.data,
            split=split_name,
            imgsz=args.imgsz,
            batch_eval=args.batch_eval,
            workers=args.workers,
        )
