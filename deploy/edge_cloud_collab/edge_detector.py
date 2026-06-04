"""
本地边缘检测模型封装 (Edge Detector)

封装 NASYOLOv8 / Ultralytics YOLO / TensorRT Engine，提供统一的单图推理接口。
支持三种后端：
  1. PyTorch (.pt) — 开发/训练阶段
  2. ONNX (.onnx) — 可选中间格式
  3. TensorRT (.engine) — Jetson 部署阶段（推荐）
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .utils import image_to_tensor, load_image, postprocess_yolo, resize_pad, scale_boxes


class EdgeDetector:
    """
    边缘端检测模型封装。

    Usage:
        # PyTorch 模式
        detector = EdgeDetector(weights="best_ema.pt")
        # TensorRT 模式（Jetson 部署）
        detector = EdgeDetector(weights="qat_int8.engine")
        result = detector.predict("path/to/image.jpg")
        # result = {
        #     "detections": [N, 6] 原始图像坐标,
        #     "latency_ms": 推理耗时,
        # }
    """

    def __init__(
        self,
        weights: str,
        device: str = "cuda:0",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        half: bool = False,
        pre_downscale: bool = True,
        pre_max_side: int = 1280,
    ):
        """
        Args:
            weights: 模型权重路径（.pt / .engine）
            device: 推理设备（仅对 PyTorch 有效）
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            imgsz: 输入尺寸
            half: 是否使用 FP16（半精度，仅 PyTorch）
            pre_downscale: 是否对过大的输入图先降采样（加速预处理）
            pre_max_side: 降采样后的最大边长
        """
        self.weights = weights
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.half = half and self.device.type != "cpu"
        self.pre_downscale = pre_downscale
        self.pre_max_side = pre_max_side

        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"模型权重不存在: {weights}")

        self.is_trt = str(weights).lower().endswith(".engine")
        self.is_onnx = str(weights).lower().endswith(".onnx")

        if self.is_trt:
            self._init_trt(weights)
            self.nc = 10
        elif self.is_onnx:
            self._init_onnx(weights)
            self.nc = 10
        else:
            self.model = self._load_model(weights)
            self.model.to(self.device).eval()
            if self.half:
                self.model = self.model.half()
            self.nc = getattr(self.model, "nc", 10)

    # ── 后端初始化 ────────────────────────────────────────────────────────────

    def _init_trt(self, engine_path: str):
        """初始化 TensorRT Runtime（Jetson 部署模式）。"""
        try:
            import sys

            # trt_runtime.py 和 edge_cloud_collab/ 同级，都在 deploy/ 下
            trt_dir = str(Path(__file__).resolve().parent.parent)
            if trt_dir not in sys.path:
                sys.path.insert(0, trt_dir)

            import trt_runtime as trt_mod

            self._trt = trt_mod.TRTRuntime(engine_path)
            self._trt_mod = trt_mod  # 缓存模块引用，避免每次 predict 重复 import
            print(f"[EdgeDetector] 加载 TensorRT Engine: {engine_path}")
        except Exception as e:
            raise RuntimeError(f"TensorRT 初始化失败: {e}")

    def _init_onnx(self, onnx_path: str):
        """初始化 ONNX Runtime（可选中间格式）。"""
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._onnx_sess = ort.InferenceSession(onnx_path, providers=providers)
            self._onnx_input_name = self._onnx_sess.get_inputs()[0].name
            print(f"[EdgeDetector] 加载 ONNX: {onnx_path}")
        except Exception as e:
            raise RuntimeError(f"ONNX 初始化失败: {e}")

    def _load_model(self, weights: str) -> torch.nn.Module:
        """加载检测模型，兼容 NASYOLOv8 和 Ultralytics YOLO。"""
        import sys

        weights_path = Path(weights)

        # 尝试 NASYOLOv8 格式
        try:
            resolved = weights_path.resolve()
            detector_dir = str(resolved.parent.parent.parent.parent)
            if detector_dir not in sys.path:
                sys.path.insert(0, detector_dir)
            from nas_yolo import NASYOLOv8

            model = NASYOLOv8.load(str(weights_path))
            print(f"[EdgeDetector] 加载 NASYOLOv8: {weights}")
            return model
        except Exception as e:
            print(f"[EdgeDetector] NASYOLOv8 加载失败: {e}")

        # 回退到 Ultralytics YOLO
        try:
            from ultralytics import YOLO

            model = YOLO(str(weights_path))
            print(f"[EdgeDetector] 加载 Ultralytics YOLO: {weights}")
            return model
        except Exception as e:
            print(f"[EdgeDetector] Ultralytics YOLO 加载失败: {e}")

        raise RuntimeError(f"无法加载模型: {weights}")

    # ── 单图推理 ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: str | Path | np.ndarray) -> dict[str, Any]:
        """
        对单张图像进行推理。

        Returns:
            dict: {
                "detections": np.ndarray [N, 6] (x1, y1, x2, y2, conf, cls),
                "latency_ms": float,
                "original_shape": (H, W),
                "input_shape": (imgsz, imgsz),
            }
        """
        if isinstance(image, (str, Path)):
            img_rgb = load_image(image)
            img_path = str(image)
        else:
            img_rgb = image
            img_path = "<ndarray>"

        orig_h, orig_w = img_rgb.shape[:2]

        if self.is_trt:
            dets, latency_ms = self._predict_trt(img_rgb)
        elif self.is_onnx:
            dets, latency_ms = self._predict_onnx(img_rgb)
        else:
            dets, latency_ms = self._predict_pytorch(img_rgb)

        return {
            "detections": dets,
            "latency_ms": latency_ms,
            "original_shape": (orig_h, orig_w),
            "input_shape": (self.imgsz, self.imgsz),
            "image_path": img_path,
        }

    # ── TensorRT 推理路径 ────────────────────────────────────────────────────

    def _predict_trt(self, img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """TensorRT 端到端推理（含预处理 + H2D + GPU + D2H + 后处理）。"""
        trt_mod = self._trt_mod

        t0 = time.perf_counter()

        # 1) 预处理：letterbox + NCHW + normalize（保证内存连续，加速 H2D）
        padded, scale, pad = resize_pad(img_rgb, self.imgsz)
        img_np = np.ascontiguousarray(padded.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0

        # 2) 推理：H2D + GPU + D2H
        self._trt.infer_async(img_np, buf_idx=0)
        self._trt.synchronize(0)
        pred_np = self._trt.get_output(0)

        # 3) 后处理：解码 + NMS + 坐标还原
        detections, _ = trt_mod.postprocess(
            pred_np,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            ratio=scale[0],
            dwdh=pad,
        )
        dets = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 6), dtype=np.float32)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return dets, latency_ms

    # ── ONNX 推理路径 ────────────────────────────────────────────────────────

    def _predict_onnx(self, img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """ONNX Runtime 推理。"""
        padded, scale, pad = resize_pad(img_rgb, self.imgsz)
        img_np = np.ascontiguousarray(padded.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0

        t0 = time.perf_counter()
        pred_np = self._onnx_sess.run(None, {self._onnx_input_name: img_np})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # ONNX 输出通常是 [1, 14, 8400]，复用 TRT 的后处理逻辑
        trt_mod = self._trt_mod

        detections, _ = trt_mod.postprocess(
            pred_np,
            conf_thres=self.conf_threshold,
            iou_thres=self.iou_threshold,
            ratio=scale[0],
            dwdh=pad,
        )
        dets = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 6), dtype=np.float32)
        return dets, latency_ms

    # ── PyTorch 推理路径 ─────────────────────────────────────────────────────

    def _predict_pytorch(self, img_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        """PyTorch 推理（NASYOLOv8 / Ultralytics）。"""
        padded, scale, pad = resize_pad(img_rgb, self.imgsz)
        tensor = image_to_tensor(padded, device=str(self.device))
        if self.half:
            tensor = tensor.half()

        t0 = time.perf_counter()

        if hasattr(self.model, "predict") and callable(getattr(self.model, "predict")):
            # Ultralytics YOLO
            results = self.model.predict(
                source=tensor,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
            )
            dets = self._parse_ultralytics_results(results)
        else:
            # NASYOLOv8
            raw_out = self.model(tensor)
            dets_list = postprocess_yolo(
                raw_out,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                imgsz=self.imgsz,
            )
            dets = dets_list[0] if len(dets_list) > 0 else np.zeros((0, 6), dtype=np.float32)
            orig_h, orig_w = img_rgb.shape[:2]
            dets = scale_boxes(dets, scale, pad, (orig_h, orig_w))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return dets, latency_ms

    # ── Ultralytics 结果解析 ────────────────────────────────────────────────

    def _parse_ultralytics_results(self, results: list) -> np.ndarray:
        """解析 Ultralytics Results 对象为统一的 [N, 6] 格式。"""
        if len(results) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        boxes = r.boxes.xyxy.cpu().numpy()  # [N, 4]
        confs = r.boxes.conf.cpu().numpy()  # [N]
        clses = r.boxes.cls.cpu().numpy()  # [N]

        dets = np.column_stack([boxes, confs, clses]).astype(np.float32)
        return dets

    # ── 预热 ────────────────────────────────────────────────────────────────

    def warmup(self, n: int = 10):
        """预热推理，消除冷启动开销（CUDA context / cache 初始化）。"""
        dummy = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(n):
            _ = self.predict(dummy)
        print(f"[EdgeDetector] Warm-up 完成: {n} iterations")

    # ── 批量推理 ────────────────────────────────────────────────────────────

    def predict_batch(self, images: list[str | Path | np.ndarray]) -> list[dict[str, Any]]:
        """批量推理（串行，简化实现）。"""
        return [self.predict(img) for img in images]

    def predict_pipeline(
        self,
        images: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """
        Double-buffer 流水线批量推理（仅 TensorRT）。
        CPU 预处理与 GPU 推理重叠，提升 throughput。

        Returns:
            list[dict]: 和 predict() 格式一致的结果列表
        """
        if not self.is_trt:
            # 非 TRT 后端回退到串行
            return self.predict_batch(images)

        trt_mod = self._trt_mod

        # ---- 阶段 1：批量预处理（CPU，可并行化） ----
        preprocessed = []
        for img_rgb in images:
            padded, scale, pad = resize_pad(img_rgb, self.imgsz)
            img_np = np.ascontiguousarray(padded.transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0
            preprocessed.append((img_np, scale, pad, img_rgb.shape[:2]))

        NUM_BUFFERS = 2
        results = [None] * len(preprocessed)

        # ---- 阶段 2：Double-buffer Pipeline ----
        t_pipeline0 = time.perf_counter()

        for i, (img_np, scale, pad, (orig_h, orig_w)) in enumerate(preprocessed):
            idx = i % NUM_BUFFERS

            # 同步取回前一个使用该 buffer 的结果
            if i >= NUM_BUFFERS:
                self._trt.synchronize(idx)
                pred_np = self._trt.get_output(idx)
                prev_i = i - NUM_BUFFERS
                _, prev_scale, prev_pad, _ = preprocessed[prev_i]
                detections, _ = trt_mod.postprocess(
                    pred_np,
                    conf_thres=self.conf_threshold,
                    iou_thres=self.iou_threshold,
                    ratio=prev_scale[0],
                    dwdh=prev_pad,
                )
                dets = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 6), dtype=np.float32)
                results[prev_i] = {
                    "detections": dets,
                    "latency_ms": -1.0,  # pipeline 模式下单帧延迟不精确
                    "original_shape": preprocessed[prev_i][3],
                    "input_shape": (self.imgsz, self.imgsz),
                    "image_path": "<pipeline>",
                }

            # 提交当前推理
            self._trt.infer_async(img_np, buf_idx=idx)

        # ---- 阶段 3：排空剩余 buffer ----
        for j in range(NUM_BUFFERS):
            idx = j % NUM_BUFFERS
            self._trt.synchronize(idx)
            pred_np = self._trt.get_output(idx)

            i = len(preprocessed) - NUM_BUFFERS + j
            if 0 <= i < len(preprocessed):
                _, scale, pad, (orig_h, orig_w) = preprocessed[i]
                detections, _ = trt_mod.postprocess(
                    pred_np,
                    conf_thres=self.conf_threshold,
                    iou_thres=self.iou_threshold,
                    ratio=scale[0],
                    dwdh=pad,
                )
                dets = np.array(detections, dtype=np.float32) if detections else np.zeros((0, 6), dtype=np.float32)
                results[i] = {
                    "detections": dets,
                    "latency_ms": -1.0,
                    "original_shape": (orig_h, orig_w),
                    "input_shape": (self.imgsz, self.imgsz),
                    "image_path": "<pipeline>",
                }

        pipeline_ms = (time.perf_counter() - t_pipeline0) * 1000.0
        avg_ms = pipeline_ms / len(preprocessed) if preprocessed else 0.0
        print(f"[EdgeDetector] Pipeline throughput: {len(preprocessed)} imgs in {pipeline_ms:.1f}ms "
              f"(avg {avg_ms:.2f}ms/img, {1000.0/avg_ms:.1f} FPS)")

        return results

    # ── 便捷工厂 ────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "EdgeDetector":
        """从配置字典实例化检测器。"""
        return cls(
            weights=cfg["weights"],
            device=cfg.get("device", "cuda:0"),
            conf_threshold=cfg.get("conf_threshold", 0.25),
            iou_threshold=cfg.get("iou_threshold", 0.45),
            imgsz=cfg.get("imgsz", 640),
            half=cfg.get("half", False),
        )
