"""CIFAR MobileNetV3 变体：分类头为「1×1 Conv + 固定核 AvgPool + 1×1 Conv」，便于 TensorRT DLA 全图部署。

与 mobilenet_v3.MobileNetV3Cifar 共享同一 backbone 与 arch_config 语义；仅头部不同，state_dict 与原版不兼容。

仅在需要 DLA 友好导出的训练脚本中引用；搜索/NAS 仍使用 MobileNetV3Cifar。"""
import torch
import torch.nn as nn

from .mbconv import MBConv, make_divisible


class MobileNetV3CifarDLAHead(nn.Module):
    """backbone 与 MobileNetV3Cifar 相同；用全卷积头替代 GAP + Linear + Flatten。

    固定输入分辨率下，final_expand 后空间尺寸 (H,W) 由 image_size 与 stride 模式唯一确定，
    使用 ``AvgPool2d(kernel_size=(H,W))`` 将特征压到 1×1（避免导出为 GlobalAveragePool REDUCE）。

    前向直接返回 ``conv_fc2`` 的 4D 张量 ``(N, num_classes, 1, 1)``，不在图末尾做 flatten/reshape。
    这样 ONNX 不含 Flatten/Reshape，TensorRT DLA 上可避免 SHUFFLE 落到 GPU；训练时在 loss 前对 logits
    做 ``squeeze``；部署/TRT 推理后对输出 ``squeeze`` 或 ``argmax(dim=1)``。
    """

    def __init__(
        self,
        arch_config,
        num_classes=10,
        dropout_rate=0.2,
        input_channel=16,
        stage_widths=None,
        last_channel=256,
        image_size=32,
    ):
        super().__init__()
        if stage_widths is None:
            stage_widths = [16, 24, 40, 80, 160]
        w_mult = arch_config.get("width_multiplier", 1.0)
        stage_widths = [make_divisible(w * w_mult) for w in stage_widths]
        input_channel = make_divisible(input_channel * w_mult)
        last_channel = make_divisible(last_channel * w_mult)

        ks = arch_config["kernel_size"]
        expand = arch_config["expand_ratio"]
        depths = arch_config["depths"]

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        )

        in_ch = input_channel
        blocks = []
        strides = [2, 2, 2, 1]
        for i, (out_ch, d) in enumerate(zip(stage_widths[1:], depths)):
            s = strides[i]
            for j in range(d):
                block_stride = s if j == 0 else 1
                blocks.append(
                    MBConv(
                        in_ch,
                        out_ch,
                        kernel_size=ks,
                        stride=block_stride,
                        expand_ratio=expand,
                        act_func="relu6",
                    )
                )
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        cexp = make_divisible(in_ch * 6)
        self.final_expand = nn.Sequential(
            nn.Conv2d(in_ch, cexp, 1, bias=False),
            nn.BatchNorm2d(cexp),
            nn.ReLU6(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            x = self.first_conv(dummy)
            x = self.blocks(x)
            x = self.final_expand(x)
            _, _, h, w = x.shape
        self._spatial = (int(h), int(w))

        # 对齐原顺序：pool 后再接「全连接」；此处用 1×1 conv + 固定核平均池化 + 1×1 conv
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv_fc1 = nn.Conv2d(cexp, last_channel, 1, bias=True)
        self.relu6 = nn.ReLU6(inplace=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.avg_pool = nn.AvgPool2d(kernel_size=self._spatial, stride=1)
        self.conv_fc2 = nn.Conv2d(last_channel, num_classes, 1, bias=True)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.final_expand(x)
        x = self.dropout1(x)
        x = self.conv_fc1(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.avg_pool(x)
        x = self.conv_fc2(x)
        # 返回 (N, C, 1, 1)，避免 ONNX 出现 Flatten/Reshape → TRT SHUFFLE 无法在 DLA 上运行。
        return x
