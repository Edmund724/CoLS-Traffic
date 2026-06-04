"""MobileNetV3 风格网络，支持从 arch_config 构建（CIFAR-10 适配，无 SE）"""
import torch.nn as nn

from .mbconv import MBConv, make_divisible


class MobileNetV3Cifar(nn.Module):
    """CIFAR-10 适配的 MobileNetV3，根据 arch_config 构建。

    arch_config: {
        "kernel_size": 3, 5, or 7,
        "expand_ratio": 2, 3, 4, 5, or 6,
        "width_multiplier": 0.75, 1.0, or 1.25,
        "depths": [d0, d1, d2, d3],  # 每 stage block 数
    }
    """

    def __init__(self, arch_config, num_classes=10, dropout_rate=0.2,
                 input_channel=16, stage_widths=None, last_channel=256):
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
                blocks.append(MBConv(
                    in_ch, out_ch,
                    kernel_size=ks,
                    stride=block_stride,
                    expand_ratio=expand,
                    act_func="relu6",
                ))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.final_ch = in_ch

        self.final_expand = nn.Sequential(
            nn.Conv2d(in_ch, make_divisible(in_ch * 6), 1, bias=False),
            nn.BatchNorm2d(make_divisible(in_ch * 6)),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(make_divisible(in_ch * 6), last_channel),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.final_expand(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
