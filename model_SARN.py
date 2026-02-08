"""
支持消融实验的 LAN 模型
添加开关控制 SA (Spatial Attention) 和 CA (Channel Attention)
"""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

#空间感知细化SARN
#Spatial_Aware_Refinement Network
class Spatial_Aware_Refinement(nn.Module):
    """
    支持消融实验的 LAN 模型

    Args:
        spatial_weight: 空间注意力权重 λ
        use_sa: 是否使用空间注意力 (SA)
        use_ca: 是否使用通道注意力 (CA)
    """

    def __init__(self, spatial_weight=0.5, use_sa=True, use_ca=True):
        super(Spatial_Aware_Refinement, self).__init__()
        self.spatial_weight = spatial_weight
        self.use_sa = use_sa
        self.use_ca = use_ca
        self.relu = nn.ReLU(inplace=True)

        number_f = 32

        # 基础卷积层
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        # 通道注意力模块 (可选)
        if use_ca:
            self.ca1 = ChannelAttention(number_f * 2)
            self.ca2 = ChannelAttention(number_f * 2)
        else:
            # 使用恒等映射替代
            self.ca1 = nn.Identity()
            self.ca2 = nn.Identity()

        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)

        # 输出层
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

        # 空间注意力模块 (可选)
        if use_sa:
            self.sa = SpatialAttention()
        else:
            # 不使用 SA 时，创建一个返回全1的模块
            self.sa = None

    def forward(self, x, gain=None):
        # 编码阶段
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        # 解码阶段 + 注意力机制
        x5_in = torch.cat([x3, x4], 1)

        # 通道注意力 (如果启用)
        if self.use_ca:
            x5_in = self.ca1(x5_in)

        x5 = self.relu(self.e_conv5(x5_in))

        x6_in = torch.cat([x2, x5], 1)

        # 通道注意力 (如果启用)
        if self.use_ca:
            x6_in = self.ca2(x6_in)

        x6 = self.relu(self.e_conv6(x6_in))

        # 生成空间注意力掩码
        if self.use_sa and self.sa is not None:
            spatial_mask = self.sa(x6)
            # 线性插值
            spatial_mask = self.spatial_weight * spatial_mask + \
                           (1 - self.spatial_weight) * torch.ones_like(spatial_mask)
        else:
            # 不使用 SA 时，掩码全为1 (无空间差异)
            spatial_mask = torch.ones((x.size(0), 1, x.size(2), x.size(3)),
                                      device=x.device)

        # 生成权重图
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r_list = torch.split(x_r, 3, dim=1)

        # 增益处理
        if gain is None:
            gain_val = torch.ones((x.size(0), 1, 1, 1)).to(x.device)
        else:
            gain_val = gain[:, 0].view(x.size(0), 1, 1, 1) if gain.dim() > 1 else gain.view(-1, 1, 1, 1)

        # 迭代增强
        curr_x = x
        for i, r in enumerate(r_list):
            dynamic_r = r * gain_val * spatial_mask
            curr_x = curr_x + dynamic_r * ((torch.pow(curr_x, 2) - curr_x) / torch.exp(curr_x))

            if i == 3:
                enhance_image_1 = curr_x

        enhance_image = curr_x
        return enhance_image_1, enhance_image, x_r


# 测试代码
if __name__ == "__main__":
    # 测试不同配置
    configs = [
        {'use_sa': True, 'use_ca': True, 'name': 'Full (SA+CA)'},
        {'use_sa': False, 'use_ca': True, 'name': 'No SA'},
        {'use_sa': True, 'use_ca': False, 'name': 'No CA'},
        {'use_sa': False, 'use_ca': False, 'name': 'No SA & CA'},
    ]

    test_input = torch.randn(1, 3, 256, 256)
    test_gain = torch.tensor([[0.5]])

    print("测试不同配置的 SARN 模型:")
    print("=" * 60)

    for cfg in configs:
        model = Spatial_Aware_Refinement(
            spatial_weight=0.5,
            use_sa=cfg['use_sa'],
            use_ca=cfg['use_ca']
        )

        _, output, _ = model(test_input, test_gain)
        print(
            f"{cfg['name']:20s} | Output shape: {output.shape} | Params: {sum(p.numel() for p in model.parameters()):,}")

    print("=" * 60)
    print("✓ 所有配置测试通过!")