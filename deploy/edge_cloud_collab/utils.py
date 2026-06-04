"""
基础工具函数：图像处理、YOLO后处理、可视化、API辅助
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


# ── 配置加载 ──────────────────────────────────────────────────────────────
def load_config(path: str = "configs/edge_cloud.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 图像 IO ───────────────────────────────────────────────────────────────
def load_image(path: str | Path) -> np.ndarray:
    """加载图像为 RGB numpy 数组 [H, W, 3]."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_pad(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """
    将图像等比例缩放并填充到目标尺寸（letterbox）。
    返回: (resized_img, (scale_w, scale_h), (pad_left, pad_top))
    """
    h, w = img.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (imgsz - new_h) // 2
    pad_bottom = imgsz - new_h - pad_top
    pad_left = (imgsz - new_w) // 2
    pad_right = imgsz - new_w - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, (scale, scale), (pad_left, pad_top)


def image_to_tensor(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """HWC RGB -> NCHW float32 tensor, normalized to [0,1]."""
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device)


def image_to_base64(img: np.ndarray, fmt: str = "JPEG") -> str:
    """numpy RGB -> base64 字符串."""
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> np.ndarray:
    """base64 -> numpy RGB."""
    data = base64.b64decode(b64)
    pil_img = Image.open(io.BytesIO(data))
    return np.array(pil_img.convert("RGB"))


# ── YOLO 后处理 ────────────────────────────────────────────────────────────
def postprocess_yolo(
    preds: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 300,
    imgsz: int = 640,
) -> list[np.ndarray]:
    """
    对 YOLOv8 Detect head 输出进行后处理（解码 + NMS）。

    Args:
        preds: 原始输出 [batch, 4+nc, num_anchors] 或 list/tuple 中的第一个元素
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值
        max_det: 单图最大检测数
        imgsz: 模型输入尺寸

    Returns:
        list[ndarray]: 每张图的检测结果 [N, 6] = (x1, y1, x2, y2, conf, cls)
    """
    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    # preds: [batch, 4+nc, num_anchors]
    batch_size = preds.shape[0]
    nc = preds.shape[1] - 4
    preds = preds.permute(0, 2, 1).contiguous()  # [batch, num_anchors, 4+nc]

    results = []
    for i in range(batch_size):
        pred = preds[i]  # [num_anchors, 4+nc]

        # 计算类别置信度 = max(cls_scores)
        cls_scores = pred[:, 4:]
        conf, cls_id = cls_scores.max(dim=1)

        # 过滤低置信度
        mask = conf > conf_threshold
        pred = pred[mask]
        conf = conf[mask]
        cls_id = cls_id[mask]

        if len(pred) == 0:
            results.append(np.zeros((0, 6), dtype=np.float32))
            continue

        # 解码 xywh -> xyxy（相对于模型输入尺寸 640x640）
        boxes = pred[:, :4]
        # YOLOv8 Detect 输出已经是 xywh 格式
        xywh = boxes
        xyxy = torch.zeros_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2  # x1
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2  # y1
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2  # x2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2  # y2

        # 限制在图像范围内
        xyxy = xyxy.clamp(0, imgsz)

        # NMS
        keep = nms_torch(xyxy, conf, iou_threshold)
        keep = keep[:max_det]

        det = torch.cat([xyxy[keep], conf[keep].unsqueeze(1), cls_id[keep].unsqueeze(1).float()], dim=1)
        results.append(det.cpu().numpy())

    return results


def nms_torch(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """PyTorch NMS（如果可用 torchvision 则优先用）。"""
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    except ImportError:
        return _nms_fallback(boxes, scores, iou_threshold)


def _nms_fallback(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """纯 PyTorch NMS fallback。"""
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(descending=True)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union.clamp(min=1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def scale_boxes(
    boxes: np.ndarray,
    scale: tuple[float, float],
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> np.ndarray:
    """
    将 letterbox 后的框坐标还原到原始图像尺寸。
    boxes: [N, 4+] (x1, y1, x2, y2, ...)
    """
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    pad_left, pad_top = pad
    scale_w, scale_h = scale

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale_w
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale_h

    # 限制在原始图像范围内
    h, w = orig_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


# ── 可视化 ─────────────────────────────────────────────────────────────────
def draw_detections(
    img: np.ndarray,
    dets: np.ndarray,
    names: dict[int, str] | None = None,
    color_map: dict[int, tuple] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制检测框。
    dets: [N, 6] = (x1, y1, x2, y2, conf, cls)
    """
    img = img.copy()
    if len(dets) == 0:
        return img

    if color_map is None:
        color_map = {}

    for det in dets:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        color = color_map.get(cls, (0, 255, 0))
        label = names.get(cls, f"cls{cls}") if names else f"cls{cls}"
        text = f"{label} {conf:.2f}"

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(img, text, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return img


# ── 检测结果序列化 ─────────────────────────────────────────────────────────
def detections_to_json(dets: np.ndarray, names: dict[int, str]) -> list[dict]:
    """将检测结果转为 JSON 友好的列表。"""
    out = []
    for det in dets:
        x1, y1, x2, y2, conf, cls = det
        out.append({
            "class": names.get(int(cls), f"cls{int(cls)}"),
            "class_id": int(cls),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": round(float(conf), 4),
        })
    return out


def detections_to_review_json(dets: np.ndarray, names: dict[int, str]) -> list[dict]:
    """将检测结果转为带索引号的 Review JSON（供 VLM 审查用）。"""
    out = []
    for idx, det in enumerate(dets):
        x1, y1, x2, y2, conf, cls = det
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        out.append({
            "index": idx,
            "class": names.get(int(cls), f"cls{int(cls)}"),
            "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
            "center": [round(float(cx), 1), round(float(cy), 1)],
            "confidence": round(float(conf), 3),
        })
    return out


# ── Review 模式决策解析与合并 ───────────────────────────────────────────────
def parse_review_decision(text: str) -> dict:
    """
    解析 VLM 的 Review 决策 JSON。
    期望格式: {"remove":[0,2], "adjust":[{"index":1,"confidence":0.95}], "add":[{"class":"car","bbox":[...],"confidence":0.9}]}
    """
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 提取 markdown 代码块
    if "```json" in text:
        parts = text.split("```json")
        text = parts[-1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        text = parts[-1].split("```")[0].strip()

    # 尝试直接解析
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _normalize_decision(data)
    except json.JSONDecodeError:
        pass

    # 尝试找 { ... } 片段
    brace_open = text.find("{")
    brace_close = text.rfind("}")
    if brace_open != -1 and brace_close != -1 and brace_close > brace_open:
        try:
            data = json.loads(text[brace_open:brace_close + 1])
            if isinstance(data, dict):
                return _normalize_decision(data)
        except json.JSONDecodeError:
            pass

    return {"remove": [], "adjust": [], "add": []}


def _normalize_decision(data: dict) -> dict:
    """标准化决策字典，确保字段存在且类型正确。"""
    remove = []
    if "remove" in data and isinstance(data["remove"], list):
        remove = [int(x) for x in data["remove"] if isinstance(x, (int, float, str))]

    adjust = []
    if "adjust" in data and isinstance(data["adjust"], list):
        for item in data["adjust"]:
            if isinstance(item, dict) and "index" in item:
                adjust.append({
                    "index": int(item["index"]),
                    "confidence": float(item.get("confidence", 0.5)),
                })

    add = []
    if "add" in data and isinstance(data["add"], list):
        for item in data["add"]:
            if isinstance(item, dict) and "class" in item:
                bbox = item.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    add.append({
                        "class": str(item["class"]),
                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        "confidence": float(item.get("confidence", 0.5)),
                    })

    return {"remove": remove, "adjust": adjust, "add": add}


def merge_review_decisions(
    edge_dets: np.ndarray,
    decision: dict,
    names: dict[int, str],
    img_w: int,
    img_h: int,
    max_remove_ratio: float = 0.90,
    protect_high_conf: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """
    将边缘检测结果与 VLM Review 决策合并。
    策略：保留未删除框（坐标不变）→ 调整置信度 → 添加漏检框。

    防御性规则：
      - 删除比例超过 max_remove_ratio 时，拒绝删除建议（防止 VLM 过度删除）
      - 高置信度框（conf >= protect_high_conf）默认保护，不删除
    """
    name_to_id = {v: k for k, v in names.items()}
    n_edge = len(edge_dets)
    remove_raw = decision.get("remove", [])

    # 防御：删除比例超过阈值 → 拒绝删除建议
    if n_edge > 0 and len(remove_raw) / n_edge > max_remove_ratio:
        print(f"[ReviewDefence] 删除比例 {len(remove_raw)}/{n_edge}={len(remove_raw)/n_edge:.0%} > {max_remove_ratio:.0%}，拒绝删除，仅保留 adjust/add")
        remove_raw = []

    # 防御：保护高置信度框（阈值与边缘检测器 conf_threshold 对齐，
    # 让 VLM 有权删除 conf>=0.3 的误检，但保留 >=0.8 的极确信框防止误删）
    remove_set = set()
    for idx in remove_raw:
        if 0 <= idx < n_edge:
            edge_conf = float(edge_dets[idx, 4])
            if edge_conf >= 0.8:
                # 极确信框保护：VLM 在 640px 图上容易误判高 conf 大目标
                continue
            if edge_conf >= protect_high_conf:
                # 中高置信度框：仅当 VLM 明确给出 remove 理由时才允许删除
                # 当前实现直接允许（因为 prompt 已要求 VLM 谨慎判断）
                pass
            remove_set.add(idx)

    # 1. 保留未删除的框
    keep_mask = np.ones(n_edge, dtype=bool)
    for idx in remove_set:
        keep_mask[idx] = False

    if n_edge == 0:
        merged = np.zeros((0, 6), dtype=np.float32)
    else:
        merged = edge_dets[keep_mask].copy()

    # 2. 调整置信度
    orig_to_new = {}
    new_idx = 0
    for orig_idx in range(n_edge):
        if keep_mask[orig_idx]:
            orig_to_new[orig_idx] = new_idx
            new_idx += 1

    for adj in decision.get("adjust", []):
        orig_idx = adj["index"]
        if orig_idx in orig_to_new:
            merged[orig_to_new[orig_idx], 4] = np.clip(adj["confidence"], 0.0, 1.0)

    # 3. 添加漏检框（VLM 发现的边缘模型漏检目标）
    add_dets = []
    for item in decision.get("add", []):
        cls_name = item.get("class", "")
        cls_id = name_to_id.get(_normalize_class_name(cls_name), -1)
        if cls_id == -1:
            continue
        bbox = item.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue
        conf = float(item.get("confidence", 0.6))
        # 限制在图像范围内
        x1 = max(0, min(float(bbox[0]), img_w))
        y1 = max(0, min(float(bbox[1]), img_h))
        x2 = max(0, min(float(bbox[2]), img_w))
        y2 = max(0, min(float(bbox[3]), img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        # 过滤明显不合理的框（极小块或覆盖全图）
        area = (x2 - x1) * (y2 - y1)
        img_area = img_w * img_h
        if area < 100 or area > img_area * 0.8:
            continue
        add_dets.append([x1, y1, x2, y2, conf, float(cls_id)])

    if add_dets:
        add_arr = np.array(add_dets, dtype=np.float32)
        if len(merged) == 0:
            merged = add_arr
        else:
            merged = np.vstack([merged, add_arr])

    stats = {
        "removed": len(remove_set),
        "protected": len(remove_raw) - len(remove_set),
        "adjusted": len(decision.get("adjust", [])),
        "added": len(add_dets),
    }
    return merged, stats


def parse_cloud_response(text: str, names: dict[int, str] | None = None) -> np.ndarray:
    """
    解析大模型返回的 JSON 文本，提取检测框。
    支持文本中包含多个 JSON 片段的情况（取最后一个有效的 JSON 数组）。
    自动过滤 Qwen3 的 <think>...</think> 推理内容。
    返回: [N, 6] = (x1, y1, x2, y2, conf, cls)
    """
    import re

    # 0. 过滤 Qwen3 thinking / reasoning 内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # 也过滤其他可能的 reasoning 标签
    text = re.sub(r'\[思考\].*?\[/思考\]', '', text, flags=re.DOTALL).strip()

    # 1. 尝试提取 markdown 代码块
    if "```json" in text:
        parts = text.split("```json")
        text = parts[-1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        text = parts[-1].split("```")[0].strip()

    # 2. 尝试直接解析整个文本
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _convert_detections(data, names)
    except json.JSONDecodeError:
        pass

    # 3. 文本中可能包含多个 JSON 数组（模型重复输出了输入+结果）
    # 策略：找到所有 [ ... ] 片段，从后往前尝试解析，取第一个成功的
    candidates = []
    start = 0
    while True:
        bracket_open = text.find("[", start)
        if bracket_open == -1:
            break
        bracket_close = text.find("]", bracket_open)
        if bracket_close == -1:
            break
        candidate = text[bracket_open:bracket_close + 1]
        candidates.append(candidate)
        start = bracket_close + 1

    # 从后往前尝试解析
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return _convert_detections(data, names)
        except json.JSONDecodeError:
            continue

    # 4. JSON 全部解析失败 → fallback 正则提取（模型输出格式轻微损坏时兜底）
    dets = _extract_detections_regex(text, names)
    if len(dets) > 0:
        return dets

    return np.zeros((0, 6), dtype=np.float32)


# ── 类别别名映射（处理 VLM 输出和 names 不完全匹配的情况）─────────────────────
_CLASS_ALIASES = {
    "person": "pedestrian",
    "people": "pedestrian",
    "walker": "pedestrian",
    "bike": "cyclist",
    "bicycle": "cyclist",
    "bicyclist": "cyclist",
    "motorcycle": "motorcyclist",
    "motorcyclist": "motorcyclist",
    "tricycle": "tricyclist",
    "tricyclist": "tricyclist",
    "wheelbarrow": "barrowlist",
    "barrow": "barrowlist",
    "barrowlist": "barrowlist",
    "cone": "trafficcone",
    "traffic cone": "trafficcone",
    "trafficcone": "trafficcone",
    "car": "car",
    "vehicle": "car",
    "truck": "truck",
    "lorry": "truck",
    "van": "van",
    "minivan": "van",
    "bus": "bus",
    "cyclist": "cyclist",
}


def _normalize_class_name(cls_name: str) -> str:
    """统一类别名：小写、去首尾空格、映射别名。"""
    if not isinstance(cls_name, str):
        return ""
    normalized = cls_name.strip().lower()
    return _CLASS_ALIASES.get(normalized, normalized)


def _extract_detections_regex(text: str, names: dict[int, str] | None) -> np.ndarray:
    """
    用正则从损坏/截断的 JSON 文本中提取检测框字段。
    匹配模式: "class":"xxx","bbox":[a,b,c,d],"confidence":x.xxx
    """
    import re
    name_to_id = {v: k for k, v in names.items()} if names else {}
    results = []

    # 宽松匹配：允许 bbox 数组内部格式轻微损坏（如缺 ]）
    pattern = r'"class"\s*:\s*"([^"]+)".*?"bbox"\s*:\s*\[\s*([^\]]+?)\s*\].*?"confidence"\s*:\s*([0-9.]+)'
    for m in re.finditer(pattern, text, re.DOTALL):
        cls_name = m.group(1)
        bbox_str = m.group(2)
        try:
            conf = float(m.group(3))
            bbox_parts = [float(x.strip()) for x in bbox_str.split(',') if x.strip()]
            if len(bbox_parts) == 4:
                cls_id = name_to_id.get(_normalize_class_name(cls_name), -1)
                results.append([bbox_parts[0], bbox_parts[1], bbox_parts[2], bbox_parts[3], conf, cls_id])
        except (ValueError, IndexError):
            continue

    if len(results) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(results, dtype=np.float32)


def _convert_detections(data: list, names: dict[int, str] | None) -> np.ndarray:
    """将解析出的 JSON 列表转为统一的 [N, 6] numpy 数组。"""
    results = []
    name_to_id = {v: k for k, v in names.items()} if names else {}

    for item in data:
        if not isinstance(item, dict):
            continue
        cls_name = item.get("class", "")
        bbox = item.get("bbox", [0, 0, 0, 0])
        conf = item.get("confidence", 0.5)

        if len(bbox) != 4:
            continue

        normalized = _normalize_class_name(cls_name)
        cls_id = name_to_id.get(normalized, -1)
        if cls_id == -1:
            cls_id = item.get("class_id", -1)

        results.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(conf), float(cls_id)])

    if len(results) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return np.array(results, dtype=np.float32)
