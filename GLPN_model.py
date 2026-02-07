import math
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn

#全局亮度感知GLPN
#Global_Luminance_Perception
class Global_Luminance_Perception(nn.Module):
    """
    亮度反馈网络（回归版）
    输入: RGB 图像 [B, 3, H, W]
    输出: 增益值 [B, 1]，范围 [0, 1]
    """

    def __init__(self, init_weights=True):
        super(Global_Luminance_Perception, self).__init__()
        self.conv1 = reflect_conv_bn(in_channels=3, out_channels=16, kernel_size=7, stride=1, pad=3)
        self.conv2 = reflect_conv_bn(in_channels=3, out_channels=16, kernel_size=5, stride=1, pad=2)
        self.conv3 = reflect_conv_bn(in_channels=3, out_channels=16, kernel_size=3, stride=1, pad=1)
        self.conv4 = reflect_conv_bn(in_channels=48, out_channels=64, kernel_size=3, stride=1, pad=1)
        self.conv5 = reflect_conv(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1)

        self.linear1 = nn.Linear(in_features=128, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=1)  # ⭐ 改为输出1个值

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        activate = nn.LeakyReLU(inplace=True)

        # 多尺度特征提取
        x1 = activate(self.conv1(x))  # 7x7
        x2 = activate(self.conv2(x))  # 5x5
        x3 = activate(self.conv3(x))  # 3x3

        # 特征融合
        x = torch.concat([x1, x2, x3], dim=1)
        x = activate(self.conv4(x))
        x = activate(self.conv5(x))

        # 全局平均池化
        x = nn.AdaptiveAvgPool2d(1)(x)  # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]

        # 全连接层
        x = self.linear1(x)
        x = activate(x)
        x = self.linear2(x)  # [B, 1]

        # Sigmoid 映射到 [0, 1]
        gain = torch.sigmoid(x)

        return gain  # [B, 1]


class reflect_conv_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad):
        super(reflect_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=pad),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=pad)
        )

    def forward(self, x):
        return self.conv(x)


# =====================================================
# 测试代码
# =====================================================
if __name__ == "__main__":
    model = Global_Luminance_Perception()

    # 测试前向传播
    test_input = torch.randn(4, 3, 256, 256)
    output = model(test_input)

    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出值范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出值: {output.squeeze().detach().numpy()}")