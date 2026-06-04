#!/usr/bin/env python3
"""
大模型协同纠错准确率提升验证脚本

测试指标：引入云端 VLM 协同纠错后，复杂场景（雨夜、遮挡等）下的 Precision@0.5 提升 > 10%

Usage:
    # 首次运行：筛选并保存复杂场景子集
    python benchmark_accuracy_coop.py \
        --engine qat_int8.engine \
        --data datasets/dair_v2x_yolo \
        --split val \
        --complexity-ratio 0.3 \
        --num 50 \
        --save-complex-scenes results/complex_scenes_val.json

    # 后续运行：直接加载已保存的子集，跳过筛选
    python benchmark_accuracy_coop.py \
        --engine qat_int8.engine \
        --data datasets/dair_v2x_yolo \
        --load-complex-scenes results/complex_scenes_val.json \
        --num 50

Output:
    - 复杂场景筛选结果（数量、典型特征）
    - 边缘-only Precision@0.5
    - 协同纠错 Precision@0.5
    - 准确率提升比例
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from edge_cloud_collab import CloudClient, EdgeDetector, ComplexityEvaluator, load_config
from edge_cloud_collab.utils import load_image


def parse_yolo_label(label_path: str, img_w: int, img_h: int):
    """解析 YOLO 格式 label [cls cx cy w h] → xyxy 像素坐标。"""
    boxes = []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    return (
        torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
    )


def scene_complexity_score(img_feats: dict, det_feats: dict) -> float:
    """综合场景复杂度分数，越高代表场景越复杂。"""
    score = 0.0

    # 阈值基于 Jetson 实测 DAIR-V2X val 集 4464 张统计标定
    # force_cloud: blur<31.4(P10), low_conf>0.50(P90), mean_conf<0.54(P10), dense>0.32(P90)

    # 1. 图像模糊（低拉普拉斯方差）
    lap = img_feats.get("laplacian_var_raw", 100)
    if lap < 31.4:
        score += 3.0
    elif lap < 50:
        score += 2.0
    elif lap < 100:
        score += 1.0

    # 2. 过曝 / 欠曝（过曝在数据集中几乎不存在，max=0.007%）
    # if img_feats.get("overexposure_ratio_raw", 0) > 0.01:
    #     score += 2.0
    if img_feats.get("underexposure_ratio_raw", 0) > 0.026:
        score += 2.0

    # 3. 目标密集（高 object_density）
    dens = det_feats.get("object_density_raw", 0)
    if dens > 0.32:
        score += 3.0
    elif dens > 0.2:
        score += 2.0
    elif dens > 0.1:
        score += 1.0

    # 4. 低置信度（模型不确定）
    if det_feats.get("mean_confidence_raw", 1.0) < 0.54:
        score += 2.0
    if det_feats.get("low_conf_ratio_raw", 0) > 0.50:
        score += 2.0

    # 5. 类别多样性（类别熵高）
    if det_feats.get("class_entropy_raw", 0) > 1.5:
        score += 1.0

    return score


def _save_complex_scenes(selected_meta, save_path):
    """保存复杂场景子集元数据到 JSON（不存大图，edge_dets 转 list）。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for item in selected_meta:
        serializable.append({
            "img_path": item["img_path"],
            "label_path": item["label_path"],
            "score": float(item["score"]),
            "img_feats": {k: float(v) for k, v in item["img_feats"].items()},
            "det_feats": {k: float(v) for k, v in item["det_feats"].items()},
            "edge_dets": item["edge_dets"].tolist(),
        })
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Save] 复杂场景子集已保存: {save_path} ({len(serializable)} 张)")


def _load_complex_scenes(load_path):
    """从 JSON 加载复杂场景子集元数据，edge_dets 转回 numpy。"""
    load_path = Path(load_path)
    with open(load_path, "r") as f:
        data = json.load(f)
    selected_meta = []
    for item in data:
        edge_dets = np.array(item["edge_dets"], dtype=np.float32)
        if edge_dets.size == 0:
            edge_dets = np.zeros((0, 6), dtype=np.float32)
        selected_meta.append({
            "img_path": item["img_path"],
            "label_path": item["label_path"],
            "score": item["score"],
            "img_feats": item["img_feats"],
            "det_feats": item["det_feats"],
            "edge_dets": edge_dets,
        })
    print(f"[Load] 复杂场景子集已加载: {load_path} ({len(selected_meta)} 张)")
    return selected_meta


def filter_complex_scenes(detector, evaluator, image_dir, label_dir, ratio=0.3, max_num=100):
    """
    筛选复杂场景子集（内存安全版）。
    流程：遍历所有图 → 边缘检测 → 评分 → **立即释放大图** → 排序 → 只加载需要的数量。
    """
    img_dir = Path(image_dir)
    lbl_dir = Path(label_dir)

    img_paths = sorted(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.jpeg"))
    )

    print(f"[Filter] 目录共 {len(img_paths)} 张图，开始评估复杂度...")
    print("[Filter] 筛选阶段只保存分数，处理完每张图立即释放内存...")

    scored_items = []
    for i, p in enumerate(img_paths):
        lbl = lbl_dir / (p.stem + ".txt")
        if not lbl.exists():
            continue

        # 加载 → 检测 → 评分 → 保存元数据 → 立即释放大图
        img_rgb = load_image(str(p))
        h, w = img_rgb.shape[:2]

        edge_result = detector.predict(img_rgb)
        edge_dets = edge_result["detections"]

        img_feats = evaluator.image_features(img_rgb)
        det_feats = evaluator.detection_features(edge_dets, (h, w))
        score = scene_complexity_score(img_feats, det_feats)

        scored_items.append({
            "img_path": str(p),
            "label_path": str(lbl),
            "score": score,
            "img_feats": img_feats,
            "det_feats": det_feats,
            "edge_dets": edge_dets.copy() if len(edge_dets) > 0 else np.zeros((0, 6), dtype=np.float32),
        })

        # 主动释放大图内存（关键！）
        del img_rgb, edge_result, edge_dets

        if (i + 1) % 100 == 0 or (i + 1) == len(img_paths):
            print(f"  已处理 {i + 1}/{len(img_paths)} 张...")

    # 按复杂度降序排列
    scored_items.sort(key=lambda x: x["score"], reverse=True)

    # 取 top ratio
    need = min(int(len(scored_items) * ratio), max_num)
    selected_meta = scored_items[:need]

    print(f"\n[Filter] 选中 {len(selected_meta)} 张复杂场景图 (top {ratio:.0%})")
    print(f"[Filter] 复杂度分数范围: {selected_meta[-1]['score']:.1f} ~ {selected_meta[0]['score']:.1f}")

    # 展示典型特征
    print("\n[Filter] 典型复杂场景特征:")
    for meta in selected_meta[:3]:
        reasons = []
        if meta["img_feats"]["laplacian_var_raw"] < 31.4:
            reasons.append("模糊")
        # 过曝在数据集中几乎不存在
        # if meta["img_feats"]["overexposure_ratio_raw"] > 0.01:
        #     reasons.append("过曝")
        if meta["img_feats"]["underexposure_ratio_raw"] > 0.026:
            reasons.append("欠曝")
        if meta["det_feats"]["object_density_raw"] > 0.32:
            reasons.append("密集")
        if meta["det_feats"]["mean_confidence_raw"] < 0.54:
            reasons.append("低置信度")
        if meta["det_feats"]["low_conf_ratio_raw"] > 0.50:
            reasons.append("低conf过多")
        print(f"  {Path(meta['img_path']).name}: score={meta['score']:.1f}, 特征={reasons}")

    print(f"\n[Filter] 选中 {len(selected_meta)} 张图（未加载大图，只保存元数据）")

    return selected_meta


def compute_prf(preds, targets, iou_thresh=0.5):
    """
    基于 IoU 匹配计算 Precision / Recall / F1-Score。
    匹配规则：按置信度降序，每个 GT 最多匹配一个 pred（同类 + IoU >= thresh）。
    """
    try:
        from torchvision.ops import box_iou
    except ImportError:
        print("[WARN] torchvision not installed, skipping PRF computation")
        return None

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        iou_matrix = box_iou(pred_boxes, gt_boxes)
        matched_gt = set()
        matched_pred = set()

        # 按置信度降序处理
        sorted_indices = torch.argsort(pred_scores, descending=True)
        for pred_idx in sorted_indices:
            pidx = pred_idx.item()
            if pidx in matched_pred:
                continue

            best_iou = 0.0
            best_gt = -1
            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                if pred_labels[pidx] != gt_labels[gt_idx]:
                    continue
                iou = iou_matrix[pidx, gt_idx].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_iou >= iou_thresh and best_gt != -1:
                total_tp += 1
                matched_gt.add(best_gt)
                matched_pred.add(pidx)
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def _nms_per_class(dets: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """类别感知的 NMS，对 numpy [N,6] 检测框做 NMS（需 torchvision）。"""
    if len(dets) == 0:
        return dets
    try:
        from torchvision.ops import nms
    except ImportError:
        return dets

    boxes_t = torch.from_numpy(dets[:, :4]).float()
    scores_t = torch.from_numpy(dets[:, 4]).float()
    labels_t = torch.from_numpy(dets[:, 5].astype(np.int64))

    # class-aware NMS：按类别偏移框坐标
    max_wh = 4096
    nms_boxes = boxes_t + labels_t.unsqueeze(1).float() * max_wh
    keep = nms(nms_boxes, scores_t, iou_thresh)
    return dets[keep.cpu().numpy()]


def evaluate_edge_only(selected_items):
    """边缘-only 推理，计算 mAP（复用筛选阶段保存的 edge_dets）。"""
    preds = []
    targets = []

    for item in selected_items:
        img_rgb = item["img_rgb"]
        h, w = img_rgb.shape[:2]
        label_path = item["label_path"]
        edge_dets = item["edge_dets"]

        # GT
        gt_boxes, gt_labels = parse_yolo_label(label_path, w, h)
        targets.append({"boxes": gt_boxes, "labels": gt_labels})

        # 复用筛选阶段的边缘检测结果
        if len(edge_dets) == 0:
            preds.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            })
        else:
            preds.append({
                "boxes": torch.from_numpy(edge_dets[:, :4]).float(),
                "scores": torch.from_numpy(edge_dets[:, 4]).float(),
                "labels": torch.from_numpy(edge_dets[:, 5].astype(np.int64)),
            })

    edge_prf = compute_prf(preds, targets, iou_thresh=0.5)
    return {"prf": edge_prf}


def evaluate_coop(cloud_client, selected_items, names):
    """
    协同纠错：VLM 输出 remove / adjust / add 决策，不修改已有框坐标。
    mAP 提升来源：precision↑（删误检）+ recall↑（补漏检）+ adjust（提升置信度）。
    """
    preds = []
    targets = []
    total_removed = 0
    total_added = 0
    total_adjusted = 0
    fallback_count = 0

    for idx, item in enumerate(selected_items):
        img_rgb = item["img_rgb"]
        h, w = img_rgb.shape[:2]
        label_path = item["label_path"]
        edge_dets = item["edge_dets"]

        # GT
        gt_boxes, gt_labels = parse_yolo_label(label_path, w, h)
        targets.append({"boxes": gt_boxes, "labels": gt_labels})

        # 云端 Review 纠错
        t0 = time.perf_counter()
        coop_result = cloud_client.correct_review(img_rgb, edge_dets, names)
        coop_dets = coop_result["detections"]
        latency = (time.perf_counter() - t0) * 1000.0
        decision = coop_result.get("decision", {})

        if not coop_result.get("success", True):
            fallback_count += 1

        n_removed = len(decision.get("remove", []))
        n_added = len(decision.get("add", []))
        n_adjusted = len(decision.get("adjust", []))
        total_removed += n_removed
        total_added += n_added
        total_adjusted += n_adjusted

        # 每张图都打印紧凑进度（Demo 可视化，避免终端长时间空白）
        changes = []
        if n_removed > 0:
            changes.append(f"删{n_removed}")
        if n_added > 0:
            changes.append(f"补{n_added}")
        if n_adjusted > 0:
            changes.append(f"调{n_adjusted}")
        change_str = f"[{','.join(changes)}]" if changes else "[无改动]"
        img_name = Path(item["img_path"]).name
        print(
            f"  [{idx + 1:>2}/{len(selected_items):>2}] {img_name:<22} "
            f"边缘{len(edge_dets):>2}→协同{len(coop_dets):>2}  "
            f"{change_str:<10} {latency:>6.0f}ms",
            flush=True,
        )

        # 协同后二次过滤：conf >= 0.3 去除低置信度误检
        if len(coop_dets) > 0:
            conf_mask = coop_dets[:, 4] >= 0.30
            coop_dets = coop_dets[conf_mask]

        # 协同后 NMS：去除 VLM add 框与保留框之间的重叠
        if len(coop_dets) > 0:
            coop_dets = _nms_per_class(coop_dets, iou_thresh=0.5)

        if len(coop_dets) == 0:
            preds.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
            })
        else:
            preds.append({
                "boxes": torch.from_numpy(coop_dets[:, :4]).float(),
                "scores": torch.from_numpy(coop_dets[:, 4]).float(),
                "labels": torch.from_numpy(coop_dets[:, 5].astype(np.int64)),
            })

    print(f"\n[Coop Stats] 总样本={len(selected_items)}, 累计删误检={total_removed}, 补漏检={total_added}, 调置信度={total_adjusted}, API 失败={fallback_count}", flush=True)
    coop_prf = compute_prf(preds, targets, iou_thresh=0.5)
    return {"prf": coop_prf}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="qat_int8.engine")
    parser.add_argument("--config", default="configs/edge_cloud.yaml")
    parser.add_argument("--data", default="datasets/dair_v2x_yolo")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--complexity-ratio", type=float, default=0.3,
                        help="复杂场景比例 (0.0-1.0)")
    parser.add_argument("--num", type=int, default=50,
                        help="最多测试多少张复杂场景图")
    parser.add_argument("--save-complex-scenes", type=str, default="",
                        help="筛选完成后保存复杂场景子集元数据到指定 JSON 路径")
    parser.add_argument("--load-complex-scenes", type=str, default="",
                        help="从 JSON 加载已保存的复杂场景子集，跳过筛选阶段")
    args = parser.parse_args()

    data_path = Path(args.data)
    image_dir = data_path / "images" / args.split
    label_dir = data_path / "labels" / args.split

    if not image_dir.exists():
        print(f"[ERROR] Image dir not found: {image_dir}")
        sys.exit(1)
    if not label_dir.exists():
        print(f"[ERROR] Label dir not found: {label_dir}")
        sys.exit(1)

    # 类别名
    names = {
        0: "car", 1: "truck", 2: "van", 3: "bus", 4: "pedestrian",
        5: "cyclist", 6: "tricyclist", 7: "motorcyclist",
        8: "barrowlist", 9: "trafficcone",
    }

    print("=" * 70, flush=True)
    print("大模型协同纠错准确率提升验证", flush=True)
    print("=" * 70, flush=True)
    print(f"Engine      : {args.engine}", flush=True)
    print(f"Data        : {args.data} / {args.split}", flush=True)
    if args.load_complex_scenes:
        print(f"Load scenes : {args.load_complex_scenes}", flush=True)
    else:
        print(f"Complexity  : top {args.complexity_ratio:.0%}", flush=True)
        print(f"Max num     : {args.num}", flush=True)
        if args.save_complex_scenes:
            print(f"Save scenes : {args.save_complex_scenes}", flush=True)
    print(f"Real API    : True (always)", flush=True)
    print("=" * 70, flush=True)

    # 1. 初始化 / 加载
    if args.load_complex_scenes:
        # 加载模式下不需要边缘检测器（edge_dets 已保存）
        print("\n[1/5] 加载模式：跳过边缘检测器初始化...")
        detector = None
        evaluator = None
    else:
        print("\n[1/5] 初始化边缘检测器...")
        detector = EdgeDetector(weights=args.engine)
        detector.warmup(n=5)
        evaluator = ComplexityEvaluator()

    # 2. 筛选 / 加载复杂场景
    print("\n[2/5] 筛选复杂场景子集...")
    if args.load_complex_scenes:
        selected_meta = _load_complex_scenes(args.load_complex_scenes)
    else:
        selected_meta = filter_complex_scenes(
            detector, evaluator,
            image_dir, label_dir,
            ratio=args.complexity_ratio,
            max_num=args.num,
        )
        if args.save_complex_scenes:
            _save_complex_scenes(selected_meta, args.save_complex_scenes)

    # 随机打乱后再切片：从全部子集中随机抽取 num 张
    import random
    random.shuffle(selected_meta)
    selected_meta = selected_meta[:args.num]
    print(f"[Filter] 已从子集中随机抽取 {len(selected_meta)} 张样本")

    # 加载选中的图片到内存
    selected = []
    for meta in selected_meta:
        img_rgb = load_image(meta["img_path"])
        selected.append({**meta, "img_rgb": img_rgb})

    if len(selected) == 0:
        print("[ERROR] 没有选中复杂场景")
        sys.exit(1)

    # 3. 边缘-only Precision
    print("\n[3/5] 计算边缘-only 识别准确率 (Precision)...", flush=True)
    edge_result = evaluate_edge_only(selected)
    if edge_result is None:
        sys.exit(1)
    edge_prf = edge_result.get("prf")
    if edge_prf:
        print(f"  Precision@0.5 : {edge_prf['precision']:.4f}", flush=True)

    # 4. 云端协同 Precision
    print("\n[4/5] 计算协同纠错识别准确率 (Precision)...", flush=True)

    cfg = load_config(args.config)
    cloud = CloudClient.from_config(cfg["cloud"], mock=False)

    coop_result = evaluate_coop(cloud, selected, names)
    if coop_result is None:
        sys.exit(1)
    coop_prf = coop_result.get("prf")
    if coop_prf:
        print(f"  Precision@0.5 : {coop_prf['precision']:.4f}", flush=True)

    # 5. 对比报告
    print("\n" + "=" * 70, flush=True)
    print("协同纠错准确率提升报告", flush=True)
    print("=" * 70, flush=True)

    # Precision 对比
    boost_p = None
    if edge_prf and coop_prf:
        base_p = edge_prf["precision"]
        coop_p = coop_prf["precision"]
        boost_p = (coop_p - base_p) / base_p * 100 if base_p > 0 else 0.0
        status_p = "✅ PASS" if boost_p > 10 else "❌ FAIL"
        print(f"  Precision@0.5:", flush=True)
        print(f"    边缘-only : {base_p:.4f} ({base_p*100:.1f}%)", flush=True)
        print(f"    协同纠错  : {coop_p:.4f} ({coop_p*100:.1f}%)", flush=True)
        print(f"    相对提升  : {boost_p:+.1f}%  {status_p}", flush=True)

    print("=" * 70, flush=True)

    # 典型样本可视化（保存前 3 张）
    print("\n[5/5] 保存典型样本对比图...")
    for idx, item in enumerate(selected[:3]):
        img_path = item["img_path"]
        img_rgb = item["img_rgb"]
        edge_dets = item["edge_dets"]
        h, w = img_rgb.shape[:2]

        # 边缘结果可视化
        vis_edge = _draw_boxes(img_rgb.copy(), edge_dets, names, color=(255, 0, 0))
        cv2.imwrite(f"coop_compare_{idx}_edge.jpg", cv2.cvtColor(vis_edge, cv2.COLOR_RGB2BGR))

        # 协同结果（Review 模式）
        coop_result = cloud.correct_review(img_rgb, edge_dets, names)
        coop_dets = coop_result["detections"]
        vis_coop = _draw_boxes(img_rgb.copy(), coop_dets, names, color=(0, 255, 0))
        cv2.imwrite(f"coop_compare_{idx}_coop.jpg", cv2.cvtColor(vis_coop, cv2.COLOR_RGB2BGR))

        print(f"  已保存 coop_compare_{idx}_edge.jpg / coop_compare_{idx}_coop.jpg")


def _draw_boxes(img: np.ndarray, dets: np.ndarray, names: dict, color=(0, 255, 0)):
    """在图上绘制检测框。"""
    if len(dets) == 0:
        return img
    for det in dets:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = names.get(int(cls), f"cls{int(cls)}")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


if __name__ == "__main__":
    main()
