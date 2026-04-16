"""MBConv 模块（无 SE）"""
import torch.nn as nn


def make_divisible(v, divisor=8):
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (无 SE)"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        expand_ratio=6,
        act_func="relu6",
    ):
        super().__init__()
        self.stride = stride
        mid_ch = make_divisible(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True) if act_func == "relu6" else nn.ReLU(inplace=True),
            ])
        layers.extend([
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride, kernel_size // 2,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True) if act_func == "relu6" else nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
