import torch
import torch.nn as nn
import torch.nn.functional as F


# 工具函数：上采样
def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)


# 工具函数：初始化权重
def initialize_weights(model):
    # 加载预训练的 MobileNetV2
    pretrained_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    pretrained_dict = pretrained_model.state_dict()

    # 筛选并匹配预训练权重
    all_params = {k: v for k, v in model.state_dict().items() if k in pretrained_dict and v.shape == pretrained_dict[k]}
    model.load_state_dict(all_params, strict=False)


# 定义基础卷积模块：卷积 + 批归一化 + 激活函数
class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 深度可分离卷积模块
class _DSConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 线性瓶颈模块（用于 MobileNetV2）
class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=1, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels * t, kernel_size=1),  # Pointwise
            _DSConv(in_channels * t, in_channels * t, stride),  # Depthwise
            nn.Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False),  # Pointwise-linear
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out += x
        return out


# 创建层的辅助函数
def _make_layer(block, in_channels, out_channels, num_blocks, t=6, stride=1):
    layers = [block(in_channels, out_channels, t, stride)]
    layers += [block(out_channels, out_channels, t, stride=1) for _ in range(1, num_blocks)]
    return nn.Sequential(*layers)

class DepthBranch(nn.Module):
    def __init__(self, **kwargs):
        super(DepthBranch, self).__init__()
        # 彩色分支（多层次特征提取）
        self.color_branch = nn.ModuleList([
            _ConvBNReLU(3, 16, kernel_size=3, stride=2, padding=1),  # 第1层
            _ConvBNReLU(16, 24, kernel_size=3, stride=2, padding=1),  # 第2层
            _ConvBNReLU(24, 32, kernel_size=3, stride=2, padding=1),  # 第3层
            _ConvBNReLU(32, 96, kernel_size=3, stride=2, padding=1),  # 第4层
            _ConvBNReLU(96, 320, kernel_size=3, stride=1, padding=1)  # 第5层
        ])

        # 深度分支（线性瓶颈模块）
        self.bottleneck1 = _make_layer(LinearBottleneck, 17, 16, num_blocks=1, t=3, stride=2)
        self.bottleneck2 = _make_layer(LinearBottleneck, 16 + 24, 24, num_blocks=3, t=3, stride=2)
        self.bottleneck3 = _make_layer(LinearBottleneck, 24 + 32, 32, num_blocks=7, t=3, stride=2)
        self.bottleneck4 = _make_layer(LinearBottleneck, 32 + 96, 96, num_blocks=3, t=2, stride=2)
        self.bottleneck5 = _make_layer(LinearBottleneck, 96 + 320, 320, num_blocks=1, t=2, stride=1)

        # 注意力机制模块
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(320, 320, kernel_size=1),
            nn.Sigmoid()
        )

        # 全局上下文特征融合
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(320, 320, kernel_size=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True)
        )

        # 最终的卷积层
        self.conv_s_d = _ConvBNReLU(320, 1, kernel_size=1, stride=1)

    def forward(self, rgb, depth):
        size = depth.size()[2:]  # 输入的特征图尺寸
        feat = []

        # 提取彩色分支的多层次特征
        color_feats = []
        color_input = rgb
        for layer in self.color_branch:
            color_input = layer(color_input)
            color_feats.append(color_input)

        # 第一阶段：融合初步彩色特征和深度特征
        x1 = torch.cat([upsample(color_feats[0], depth.size()[2:]), depth], dim=1)
        x1 = self.bottleneck1(x1)
        feat.append(x1)

        # 第二阶段到第五阶段：逐阶段融合特征
        stages = [self.bottleneck2, self.bottleneck3, self.bottleneck4, self.bottleneck5]
        for i, bottleneck in enumerate(stages):
            x1 = torch.cat([upsample(color_feats[i + 1], x1.size()[2:]), x1], dim=1)
            x1 = bottleneck(x1)
            feat.append(x1)

        # 全局上下文融合
        global_feat = self.global_context(x1)
        x1 = x1 + global_feat

        # 注意力机制
        attention_map = self.attention(x1)
        x1 = x1 * attention_map

        # 最终生成深度图
        s_d = self.conv_s_d(x1)

        return s_d, feat
