from __future__ import annotations

import csv
import time
from pathlib import Path

import torch


def count_params_m(module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def count_trainable_params_m(module) -> float:
    return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6


def append_metrics_row(csv_path: str | Path, row: dict) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def upsert_summary_row(csv_path: str | Path, row: dict, key_field: str = "run_dir") -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    fieldnames = list(row.keys())

    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames or []
            for existing_row in reader:
                rows.append(existing_row)
        for name in existing_fieldnames:
            if name not in fieldnames:
                fieldnames.append(name)

    updated = False
    normalized_row = {name: row.get(name, "") for name in fieldnames}
    for idx, existing_row in enumerate(rows):
        if existing_row.get(key_field) == str(row.get(key_field, "")):
            rows[idx] = {name: normalized_row.get(name, existing_row.get(name, "")) for name in fieldnames}
            updated = True
            break

    if not updated:
        rows.append(normalized_row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{name: r.get(name, "") for name in fieldnames} for r in rows])


def benchmark_forward_fps(
    loader,
    device,
    prepare_batch_fn,
    forward_fn,
    max_batches: int = 30,
    warmup_batches: int = 5,
):
    if max_batches <= 0:
        return None, None, 0, 0

    was_training = getattr(forward_fn, "__self__", None)
    if was_training is not None and hasattr(was_training, "training"):
        was_training_mode = was_training.training
    else:
        was_training_mode = None

    total_images = 0
    measured_batches = 0
    total_time = 0.0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            batch = prepare_batch_fn(batch_data)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            _ = forward_fn(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            if batch_idx >= warmup_batches:
                total_time += elapsed
                total_images += int(batch["img"].shape[0])
                measured_batches += 1

            if batch_idx + 1 >= warmup_batches + max_batches:
                break

    if was_training_mode is not None:
        was_training.train(was_training_mode)

    if total_time <= 0 or total_images == 0:
        return None, None, measured_batches, total_images

    fps = total_images / total_time
    latency_ms = (total_time / total_images) * 1000.0
    return fps, latency_ms, measured_batches, total_images
