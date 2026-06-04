#!/usr/bin/env python3
"""
TensorRT End-to-End Runtime with DOUBLE-BUFFER pipeline.
Compatible with both legacy API (<8.5) and new API (8.5+/10.x).

Usage:
    python trt_runtime.py --engine qat_int8.engine --input test.jpg --save
    python trt_runtime.py --engine qat_int8.engine --input test.jpg --benchmark 1000
    python trt_runtime.py --engine qat_int8.engine --input test.jpg --benchmark-pipe 1000
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    _HAS_TRT = True
except Exception as e:
    _HAS_TRT = False
    print(f"[WARN] TensorRT / pycuda not available: {e}")

CONF_THRES = 0.25
IOU_THRES = 0.45
INPUT_SIZE = 640
NC = 10
CLASS_NAMES = [
    "car", "truck", "van", "bus", "pedestrian",
    "cyclist", "tricyclist", "motorcyclist", "barrowlist", "trafficcone",
]
NUM_BUFFERS = 2


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


# 预分配预处理输出 buffer（可复用，减少 malloc 开销）
_PREPROC_BUF = {}


def _get_preproc_buf(img_size: int):
    """获取或创建预处理输出 buffer。"""
    if img_size not in _PREPROC_BUF:
        _PREPROC_BUF[img_size] = np.empty((1, 3, img_size, img_size), dtype=np.float32)
    return _PREPROC_BUF[img_size]


def preprocess(img_path: str | np.ndarray, img_size: int = 640, buf: np.ndarray | None = None):
    """
    预处理：letterbox + BGR→RGB + NCHW + normalize。
    支持图片路径或 numpy BGR 数组输入。
    如果传入 buf，直接写入 buf 避免额外内存分配。
    """
    t0 = time.perf_counter()
    if isinstance(img_path, str):
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            raise ValueError(f"Cannot read image: {img_path}")
    else:
        img0 = img_path

    img, ratio, dwdh = letterbox(img0, (img_size, img_size), auto=False)

    # 一步完成：BGR→RGB, HWC→CHW, contiguous, float32, normalize
    if buf is not None:
        buf[0] = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        img_np = buf
    else:
        img_np = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1)[None, ...], dtype=np.float32) / 255.0

    t1 = time.perf_counter()
    return img_np, img0, ratio, dwdh, (t1 - t0) * 1000.0


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """纯 numpy NMS，避免 torch CPU→GPU 搬运（小批量时更快）。"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess(pred_np: np.ndarray, conf_thres=0.25, iou_thres=0.45, ratio=1.0, dwdh=(0, 0), use_torch_nms=True):
    """
    后处理：解码 + NMS + 坐标还原。
    use_torch_nms=True 用 torchvision GPU NMS（框多时快）；
    use_torch_nms=False 用 numpy NMS（框少时省搬运开销）。
    """
    t0 = time.perf_counter()

    # 形状兼容
    if pred_np.shape[1] == 14 and pred_np.shape[2] == 8400:
        pass
    elif pred_np.shape[1] == 8400 and pred_np.shape[2] == 14:
        pred_np = pred_np.transpose(0, 2, 1)
    else:
        raise ValueError(f"Unexpected pred shape: {pred_np.shape}")

    pred = pred_np[0]  # [14, 8400]
    boxes_xywh = pred[:4].T  # [8400, 4]
    cls_scores = pred[4:].T  # [8400, nc]

    # 解码 xywh → xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 类别置信度
    scores = cls_scores.max(axis=1)
    labels = cls_scores.argmax(axis=1)

    # 过滤低置信度
    mask = scores > conf_thres
    boxes_xyxy = boxes_xyxy[mask]
    scores = scores[mask]
    labels = labels[mask]

    # NMS
    if len(scores) > 0:
        if use_torch_nms:
            # GPU NMS（框数量多时更快）
            boxes_t = torch.from_numpy(boxes_xyxy)
            scores_t = torch.from_numpy(scores)
            labels_t = torch.from_numpy(labels)
            max_wh = INPUT_SIZE + 1
            nms_boxes = boxes_t + labels_t.unsqueeze(1).float() * max_wh
            keep = torchvision.ops.nms(nms_boxes, scores_t, iou_thres)
            boxes_xyxy = boxes_xyxy[keep.cpu().numpy()]
            scores = scores[keep.cpu().numpy()]
            labels = labels[keep.cpu().numpy()]
        else:
            # numpy NMS（框数量少时省 CPU→GPU 搬运）
            keep = _nms_numpy(boxes_xyxy, scores, iou_thres)
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            labels = labels[keep]

    # 坐标还原
    detections = []
    left, top = dwdh
    for (x1, y1, x2, y2), s, l in zip(boxes_xyxy, scores, labels):
        detections.append([
            (x1 - left) / ratio,
            (y1 - top) / ratio,
            (x2 - left) / ratio,
            (y2 - top) / ratio,
            float(s),
            int(l),
        ])

    t1 = time.perf_counter()
    return detections, (t1 - t0) * 1000.0


def draw_detections(img, detections, class_names):
    for x1, y1, x2, y2, conf, cls_id in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{class_names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


class _Buffer:
    """Per-context buffer pool (all inputs + all outputs)."""
    def __init__(self):
        self.inputs = {}       # name -> info
        self.outputs = {}      # name -> info
        self.bindings = []     # legacy API: list of device ptrs in binding order
        self.first_input_name = None
        self.first_output_name = None


class TRTRuntime:
    """TensorRT runtime with double-buffer pipeline."""

    def __init__(self, engine_path: str, num_buffers: int = NUM_BUFFERS):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.use_new_api = hasattr(self.engine, "num_io_tensors")
        self.num_buffers = num_buffers

        self.streams = [cuda.Stream() for _ in range(num_buffers)]
        self.contexts = [self.engine.create_execution_context() for _ in range(num_buffers)]
        self.buffers = []

        for _ in range(num_buffers):
            buf = self._alloc_buffer()
            self.buffers.append(buf)

        print(f"[TRT] Engine loaded: {engine_path}")
        print(f"[TRT] API mode: {'new (8.5+)' if self.use_new_api else 'legacy (<8.5)'}")
        print(f"[TRT] Buffers: {num_buffers}")
        b0 = self.buffers[0]
        print(f"[TRT] Inputs : {list(b0.inputs.keys())}")
        print(f"[TRT] Outputs: {list(b0.outputs.keys())}")
        first_in = b0.inputs[b0.first_input_name]
        first_out = b0.outputs[b0.first_output_name]
        print(f"[TRT] Input shape : {first_in['shape']}")
        print(f"[TRT] Output shape: {first_out['shape']}")

    def _alloc_buffer(self):
        buf = _Buffer()

        if self.use_new_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                shape = self.engine.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = trt.volume(shape)
                nbytes = size * np.dtype(dtype).itemsize

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(nbytes)

                d = {"name": name, "host": host_mem, "device": device_mem,
                     "shape": shape, "dtype": dtype, "nbytes": nbytes}

                if mode == trt.TensorIOMode.INPUT:
                    buf.inputs[name] = d
                    if buf.first_input_name is None:
                        buf.first_input_name = name
                else:
                    buf.outputs[name] = d
                    if buf.first_output_name is None:
                        buf.first_output_name = name
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = self.engine.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                size = trt.volume(shape)
                nbytes = size * np.dtype(dtype).itemsize

                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(nbytes)
                buf.bindings.append(int(device_mem))

                d = {"name": name, "host": host_mem, "device": device_mem,
                     "shape": shape, "dtype": dtype, "nbytes": nbytes}

                if self.engine.binding_is_input(i):
                    buf.inputs[name] = d
                    if buf.first_input_name is None:
                        buf.first_input_name = name
                else:
                    buf.outputs[name] = d
                    if buf.first_output_name is None:
                        buf.first_output_name = name

        return buf

    def infer_async(self, input_np: np.ndarray, buf_idx: int = 0):
        buf = self.buffers[buf_idx]
        stream = self.streams[buf_idx]
        ctx = self.contexts[buf_idx]

        # H2D: first input
        in_name = buf.first_input_name
        in_info = buf.inputs[in_name]
        np.copyto(in_info["host"], input_np.ravel())
        cuda.memcpy_htod_async(in_info["device"], in_info["host"], stream)

        # GPU
        if self.use_new_api:
            for info in buf.inputs.values():
                ctx.set_tensor_address(info["name"], int(info["device"]))
            for info in buf.outputs.values():
                ctx.set_tensor_address(info["name"], int(info["device"]))

            # Dynamic batch handling
            current_shape = list(in_info["shape"])
            if input_np.shape[0] != current_shape[0] and current_shape[0] == -1:
                ctx.set_input_shape(in_name, input_np.shape)
                for out_name, out_info in buf.outputs.items():
                    new_shape = ctx.get_tensor_shape(out_name)
                    if new_shape != out_info["shape"]:
                        size = trt.volume(new_shape)
                        dtype = out_info["dtype"]
                        out_info["host"] = cuda.pagelocked_empty(size, dtype)
                        out_info["device"] = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
                        out_info["shape"] = new_shape
            ctx.execute_async_v3(stream_handle=stream.handle)
        else:
            # Update all bindings with current buffer's device pointers
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    ptr = int(buf.inputs[name]["device"])
                else:
                    ptr = int(buf.outputs[name]["device"])
                buf.bindings[i] = ptr
            ctx.execute_async_v2(bindings=buf.bindings, stream_handle=stream.handle)

        # D2H: first output
        out_name = buf.first_output_name
        out_info = buf.outputs[out_name]
        cuda.memcpy_dtoh_async(out_info["host"], out_info["device"], stream)
        return buf

    def synchronize(self, buf_idx: int = 0):
        self.streams[buf_idx].synchronize()

    def get_output(self, buf_idx: int = 0):
        buf = self.buffers[buf_idx]
        out_info = buf.outputs[buf.first_output_name]
        return out_info["host"].reshape(out_info["shape"])


def _probe_inference_breakdown(runtime: TRTRuntime, img_np: np.ndarray):
    """Measure H2D / GPU / D2H by instrumenting a single buffer."""
    buf = runtime.buffers[0]
    stream = runtime.streams[0]
    ctx = runtime.contexts[0]
    in_info = buf.inputs[buf.first_input_name]
    out_info = buf.outputs[buf.first_output_name]

    h2d_times = []
    gpu_times = []
    d2h_times = []

    for _ in range(100):
        # H2D: 计时仅包含 launch 开销（async）
        t0 = time.perf_counter()
        np.copyto(in_info["host"], img_np.ravel())
        cuda.memcpy_htod_async(in_info["device"], in_info["host"], stream)
        t1 = time.perf_counter()
        h2d_times.append((t1 - t0) * 1000.0)

        # 确保 H2D 完成后再计时纯 GPU 执行，避免 H2D 等待时间污染 GPU 指标
        stream.synchronize()

        if runtime.use_new_api:
            for info in buf.inputs.values():
                ctx.set_tensor_address(info["name"], int(info["device"]))
            for info in buf.outputs.values():
                ctx.set_tensor_address(info["name"], int(info["device"]))
            t0 = time.perf_counter()
            ctx.execute_async_v3(stream_handle=stream.handle)
            stream.synchronize()
            t1 = time.perf_counter()
            gpu_times.append((t1 - t0) * 1000.0)
        else:
            for i in range(runtime.engine.num_bindings):
                name = runtime.engine.get_binding_name(i)
                ptr = int(buf.inputs[name]["device"]) if runtime.engine.binding_is_input(i) else int(buf.outputs[name]["device"])
                buf.bindings[i] = ptr
            t0 = time.perf_counter()
            ctx.execute_async_v2(bindings=buf.bindings, stream_handle=stream.handle)
            stream.synchronize()
            t1 = time.perf_counter()
            gpu_times.append((t1 - t0) * 1000.0)

        # D2H: 计时包含 launch 到完成（因需等待 GPU 结果）
        t0 = time.perf_counter()
        cuda.memcpy_dtoh_async(out_info["host"], out_info["device"], stream)
        stream.synchronize()
        t1 = time.perf_counter()
        d2h_times.append((t1 - t0) * 1000.0)

    return np.array(h2d_times), np.array(gpu_times), np.array(d2h_times)


def benchmark_sync(runtime: TRTRuntime, img0: np.ndarray, n_runs: int = 1000, warmup: int = 50):
    print(f"[Bench-Sync] Warm-up {warmup}...")
    preproc_buf = _get_preproc_buf(INPUT_SIZE)
    for _ in range(warmup):
        img, ratio, dwdh = letterbox(img0, (INPUT_SIZE, INPUT_SIZE), auto=False)
        preproc_buf[0] = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        runtime.infer_async(preproc_buf, buf_idx=0)
        runtime.synchronize(0)
        _ = postprocess(runtime.get_output(0), CONF_THRES, IOU_THRES, ratio, dwdh)

    print(f"[Bench-Sync] Running {n_runs} iterations...")
    preprocess_times = []
    postprocess_times = []
    total_times = []

    img, ratio, dwdh = letterbox(img0, (INPUT_SIZE, INPUT_SIZE), auto=False)
    preproc_buf[0] = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0

    h2d_times, gpu_times, d2h_times = _probe_inference_breakdown(runtime, preproc_buf)

    for i in range(n_runs):
        t0 = time.perf_counter()
        img, ratio, dwdh = letterbox(img0, (INPUT_SIZE, INPUT_SIZE), auto=False)
        preproc_buf[0] = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        t1 = time.perf_counter()
        preprocess_times.append((t1 - t0) * 1000.0)

        t_total0 = time.perf_counter()
        runtime.infer_async(preproc_buf, buf_idx=0)
        runtime.synchronize(0)
        t_total1 = time.perf_counter()
        total_times.append((t_total1 - t_total0) * 1000.0)

        out = runtime.get_output(0)
        _, t_post = postprocess(out, CONF_THRES, IOU_THRES, ratio, dwdh)
        postprocess_times.append(t_post)

    preprocess_times = np.array(preprocess_times)
    postprocess_times = np.array(postprocess_times)
    total_times = np.array(total_times)
    e2e_times = preprocess_times + total_times + postprocess_times

    print("\n" + "=" * 75)
    print("SYNC Benchmark (Single Stream)")
    print("=" * 75)
    print(f"{'Stage':<18} {'Mean(ms)':>10} {'Std(ms)':>10} {'P50':>10} {'P90':>10} {'P99':>10} {'Ratio':>8}")
    print("-" * 75)
    for name, arr in [
        ("preprocess", preprocess_times),
        ("H2D", h2d_times),
        ("GPU", gpu_times),
        ("D2H", d2h_times),
        ("postprocess", postprocess_times),
        ("inference_only", total_times),
        ("end2end", e2e_times),
    ]:
        ratio_pct = arr.mean() / e2e_times.mean() * 100 if e2e_times.mean() > 0 else 0
        print(f"{name:<18} {arr.mean():>10.3f} {arr.std():>10.3f} {np.percentile(arr, 50):>10.3f} {np.percentile(arr, 90):>10.3f} {np.percentile(arr, 99):>10.3f} {ratio_pct:>7.1f}%")
    print("-" * 75)
    print(f"End-to-end FPS    : {1000.0 / e2e_times.mean():.2f}")
    print(f"Inference-only FPS: {1000.0 / total_times.mean():.2f}")
    print("=" * 75)
    return e2e_times.mean(), total_times.mean()


def benchmark_pipeline(runtime: TRTRuntime, img0: np.ndarray, n_runs: int = 1000, warmup: int = 50):
    print(f"[Bench-Pipe] Warm-up {warmup}...")
    preproc_buf_pipe = _get_preproc_buf(INPUT_SIZE)
    for i in range(warmup):
        idx = i % NUM_BUFFERS
        img, ratio, dwdh = letterbox(img0, (INPUT_SIZE, INPUT_SIZE), auto=False)
        preproc_buf_pipe[0] = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        if i >= NUM_BUFFERS:
            runtime.synchronize(idx)
        runtime.infer_async(preproc_buf_pipe, buf_idx=idx)
    for j in range(NUM_BUFFERS):
        runtime.synchronize(j)

    # 额外测一次纯 GPU 分解时间（供模块四总结表 <30ms 判定使用）
    print("[Bench-Pipe] Probing GPU breakdown...")
    h2d_times, gpu_times, d2h_times = _probe_inference_breakdown(runtime, preproc_buf_pipe)

    print(f"[Bench-Pipe] Running {n_runs} iterations (double-buffer)...")
    frame_intervals = []
    preprocess_times = []
    postprocess_times = []

    _, ratio, dwdh = letterbox(img0, (INPUT_SIZE, INPUT_SIZE), auto=False)

    t_start = time.perf_counter()
    for i in range(n_runs):
        idx = i % NUM_BUFFERS

        t0 = time.perf_counter()
        img, _, _ = letterbox(img0, (INPUT_SIZE, INPUT_SIZE), auto=False)
        preproc_buf_pipe[0] = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
        t1 = time.perf_counter()
        preprocess_times.append((t1 - t0) * 1000.0)

        if i >= NUM_BUFFERS:
            t_wait0 = time.perf_counter()
            runtime.synchronize(idx)
            t_wait1 = time.perf_counter()
            out = runtime.get_output(idx)
            t_p0 = time.perf_counter()
            _, _ = postprocess(out, CONF_THRES, IOU_THRES, ratio, dwdh)
            t_p1 = time.perf_counter()
            postprocess_times.append((t_p1 - t_p0) * 1000.0)
            frame_intervals.append((t_wait1 - t_start) * 1000.0)
            t_start = t_wait1

        runtime.infer_async(preproc_buf_pipe, buf_idx=idx)

    for j in range(NUM_BUFFERS):
        runtime.synchronize(j)
        out = runtime.get_output(j)
        t_p0 = time.perf_counter()
        _, _ = postprocess(out, CONF_THRES, IOU_THRES, ratio, dwdh)
        t_p1 = time.perf_counter()
        postprocess_times.append((t_p1 - t_p0) * 1000.0)

    frame_intervals = np.array(frame_intervals[1:])
    preprocess_times = np.array(preprocess_times)
    postprocess_times = np.array(postprocess_times)

    print("\n" + "=" * 75)
    print("PIPELINE Benchmark (Double-Buffer, CPU-GPU Overlap)")
    print("=" * 75)
    print(f"{'Metric':<25} {'Mean(ms)':>10} {'P50':>10} {'P90':>10} {'P99':>10}")
    print("-" * 75)
    print(f"{'Frame interval':<25} {frame_intervals.mean():>10.3f} {np.percentile(frame_intervals, 50):>10.3f} {np.percentile(frame_intervals, 90):>10.3f} {np.percentile(frame_intervals, 99):>10.3f}")
    print(f"{'Preprocess (CPU)':<25} {preprocess_times.mean():>10.3f} {np.percentile(preprocess_times, 50):>10.3f} {np.percentile(preprocess_times, 90):>10.3f} {np.percentile(preprocess_times, 99):>10.3f}")
    print(f"{'Postprocess (CPU)':<25} {postprocess_times.mean():>10.3f} {np.percentile(postprocess_times, 50):>10.3f} {np.percentile(postprocess_times, 90):>10.3f} {np.percentile(postprocess_times, 99):>10.3f}")
    print("-" * 75)
    print(f"Pipeline FPS      : {1000.0 / frame_intervals.mean():.2f}")
    # 简洁输出纯 GPU 分解时间，供 benchmark_compare.py 解析
    print(f"[GPU-Breakdown] mean={gpu_times.mean():.3f}ms std={gpu_times.std():.3f}ms p90={np.percentile(gpu_times, 90):.3f}ms")
    print("=" * 75)
    return frame_intervals.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output.jpg")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--benchmark", type=int, default=0, help="Run N iterations sync benchmark")
    parser.add_argument("--benchmark-pipe", type=int, default=0, help="Run N iterations pipeline benchmark")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if not _HAS_TRT:
        sys.exit(1)

    runtime = TRTRuntime(args.engine)
    input_path = Path(args.input)

    if args.benchmark > 0 or args.benchmark_pipe > 0:
        if not input_path.is_file():
            print("[ERROR] Benchmark requires a single image file")
            sys.exit(1)
        img0 = cv2.imread(str(input_path))
        if img0 is None:
            print(f"[ERROR] Cannot read image: {input_path}")
            sys.exit(1)

        if args.benchmark > 0:
            benchmark_sync(runtime, img0, n_runs=args.benchmark)
        if args.benchmark_pipe > 0:
            benchmark_pipeline(runtime, img0, n_runs=args.benchmark_pipe)
        return

    if input_path.is_file():
        image_paths = [input_path]
        out_dir = Path(args.output).parent if args.save else None
    else:
        image_paths = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
        out_dir = Path(args.output) if args.save else None
        if args.save and out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img_np, img0, ratio, dwdh, t_pre = preprocess(str(img_path))
        t0 = time.perf_counter()
        runtime.infer_async(img_np, buf_idx=0)
        runtime.synchronize(0)
        t1 = time.perf_counter()
        t_infer = (t1 - t0) * 1000.0

        out = runtime.get_output(0)
        detections, t_post = postprocess(out, args.conf, args.iou, ratio, dwdh)
        total_ms = t_pre + t_infer + t_post
        print(f"[{img_path.name}] pre={t_pre:.2f} infer={t_infer:.2f} post={t_post:.2f} | total={total_ms:.2f}ms | {len(detections)} dets")
        if args.save:
            img_vis = draw_detections(img0.copy(), detections, CLASS_NAMES)
            out_path = out_dir / img_path.name if out_dir else Path(args.output)
            cv2.imwrite(str(out_path), img_vis)
            print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
