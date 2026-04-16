#!/usr/bin/env python3
"""将 PyTorch 模型导出为 ONNX（支持标准 NAS 模型与 DLA 友好头模型）

DLA 头模型（MobileNetV3CifarDLAHead）当前导出为 logits 形状 (N, C, 1, 1)，无 Flatten/Reshape，
便于 TensorRT DLA 全图部署；Jetson 推理时对输出做 squeeze 或 argmax(dim=1)。

用法:
  python export_onnx.py
  python export_onnx.py --model path/to/best_model.pth
  python export_onnx.py --model fixed_dla_xxx.pth -o out.onnx
  python export_onnx.py --model xxx.pth --dynamic
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_fbnet.models import MobileNetV3Cifar
from nas_fbnet.models.mobilenet_v3_dla import MobileNetV3CifarDLAHead
from nas_fbnet.config import (
    STAGE_WIDTHS,
    INPUT_CHANNEL,
    LAST_CHANNEL,
    NUM_CLASSES,
    BEST_MODEL_PATH,
    OUTPUT_DIR,
)


def _detect_model_type(ckpt, state_dict):
    """根据 checkpoint 字段或权重键名选择模型类。"""
    t = ckpt.get("model_type")
    if t == "MobileNetV3CifarDLAHead":
        return "dla"
    if t == "MobileNetV3Cifar" or t is None:
        if any(k.startswith("conv_fc") for k in state_dict.keys()):
            return "dla"
        return "standard"
    # 未知字符串时按权重推断
    if any(k.startswith("conv_fc") for k in state_dict.keys()):
        return "dla"
    return "standard"


def load_model(model_path, image_size=None):
    """从 .pth 加载模型（内嵌 config + state_dict）。

    image_size: 仅 DLA 头模型需要；None 时优先用 checkpoint 里的 image_size，否则 32。
    返回 (model, arch_config, export_meta)，export_meta 含 image_size 用于 dummy 输入形状。
    """
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "config" in ckpt:
        arch_config = ckpt["config"].copy()
        for k in ["kernel_size", "expand_ratio", "depths"]:
            if k not in arch_config:
                raise ValueError(f"模型 config 缺少字段: {k}")
        state_dict = ckpt["state_dict"]
    else:
        raise ValueError(
            "模型格式错误：需包含 config 和 state_dict。请使用搜索/重训练/ train_dla 保存的 .pth 文件。"
        )

    kind = _detect_model_type(ckpt, state_dict)
    img_sz = image_size if image_size is not None else ckpt.get("image_size", 32)

    if kind == "dla":
        model = MobileNetV3CifarDLAHead(
            arch_config,
            num_classes=NUM_CLASSES,
            dropout_rate=0.0,
            input_channel=INPUT_CHANNEL,
            stage_widths=STAGE_WIDTHS,
            last_channel=LAST_CHANNEL,
            image_size=img_sz,
        )
    else:
        model = MobileNetV3Cifar(
            arch_config,
            num_classes=NUM_CLASSES,
            dropout_rate=0.0,
            input_channel=INPUT_CHANNEL,
            stage_widths=STAGE_WIDTHS,
            last_channel=LAST_CHANNEL,
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    logits_4d = bool(kind == "dla" and ckpt.get("logits_4d", True))
    meta = {"model_kind": kind, "image_size": img_sz, "logits_4d": logits_4d}
    return model, arch_config, meta


def export_onnx(model, output_path, input_shape=(1, 3, 32, 32), dynamic_batch=False, opset_version=14):
    """导出模型为 ONNX。"""
    dummy_input = torch.randn(*input_shape)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )


def main():
    default_onnx_dir = os.path.join(OUTPUT_DIR, "onnx_models")
    parser = argparse.ArgumentParser(description="导出 NAS-FBNet / DLA 头模型为 ONNX")
    parser.add_argument("--model", default='./nas_fbnet_outputs/fixed_dla_ks3_e6_w1.25_d4422_1.7M_test0.85.pth', help="输入 .pth 模型路径")
    parser.add_argument("-o", "--output", type=str, help="输出 .onnx 完整路径（优先于 --output-dir）")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(OUTPUT_DIR, "onnx"),
        help=f"ONNX 保存目录（默认: {default_onnx_dir}）",
    )
    parser.add_argument("--dynamic", action="store_true", help="启用动态 batch 维度")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset 版本")
    parser.add_argument("--batch-size", type=int, default=1, help="导出时用于 shape 推导的 batch 大小")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="输入 H=W（CIFAR 为 32）；仅 DLA 头模型需要与训练一致，默认读 checkpoint",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"未找到模型: {args.model}")
        sys.exit(1)

    if args.output:
        output_path = args.output
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        basename = os.path.basename(args.model)
        name = os.path.splitext(basename)[0] + ".onnx"
        output_path = os.path.join(args.output_dir, name)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"加载模型: {args.model}")
    model, arch_config, meta = load_model(args.model, image_size=args.image_size)
    print(f"架构: {arch_config}")
    print(f"导出类型: {meta['model_kind']} (image_size={meta['image_size']})")
    if meta.get("logits_4d"):
        print("ONNX 输出形状: (N, num_classes, 1, 1) — TRT/后处理请 squeeze 或 argmax(dim=1)。")

    h = meta["image_size"]
    input_shape = (args.batch_size, 3, h, h)
    print(f"导出 ONNX: {output_path} (input_shape={input_shape}, dynamic_batch={args.dynamic})")
    export_onnx(
        model,
        output_path,
        input_shape=input_shape,
        dynamic_batch=args.dynamic,
        opset_version=args.opset,
    )
    print("完成。")


if __name__ == "__main__":
    main()
