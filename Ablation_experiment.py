"""
基于您原有训练脚本的 SA/CA 消融实验训练
训练4个版本:
1. Full (SA + CA) - 完整模型
2. No SA - 仅移除空间注意力
3. No CA - 仅移除通道注意力
4. No SA & CA - 同时移除两者
"""

import os
import argparse
import random
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from math import exp
import torchvision.models as models

from GLPN_model import Global_Luminance_Perception
from SARN_model import Spatial_Aware_Refinement  # ⭐ 使用消融版本


# =====================================================
# 数据集
# =====================================================
class PairwiseDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.low_path = os.path.join(root_path, 'low')
        self.high_path = os.path.join(root_path, 'hight')

        if not os.path.exists(self.low_path):
            raise FileNotFoundError(f"低光目录不存在: {self.low_path}")
        if not os.path.exists(self.high_path):
            raise FileNotFoundError(f"正常光目录不存在: {self.high_path}")

        self.file_names = sorted(os.listdir(self.low_path))
        self.transform = transform

        print(f"📂 加载数据集: {len(self.file_names)} 对图像")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        low_img = Image.open(os.path.join(self.low_path, name)).convert('RGB')
        high_img = Image.open(os.path.join(self.high_path, name)).convert('RGB')

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


# =====================================================
# SSIM 损失
# =====================================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


# =====================================================
# Perceptual Loss（感知损失）
# =====================================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))
        self.slice2 = nn.Sequential(*list(vgg[4:9]))
        self.slice3 = nn.Sequential(*list(vgg[9:16]))

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        pred_f1 = self.slice1(pred)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)

        target_f1 = self.slice1(target)
        target_f2 = self.slice2(target_f1)
        target_f3 = self.slice3(target_f2)

        loss = (F.l1_loss(pred_f1, target_f1) +
                F.l1_loss(pred_f2, target_f2) +
                F.l1_loss(pred_f3, target_f3))

        return loss


# =====================================================
# Gradient Loss（梯度损失）
# =====================================================
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        return loss


# =====================================================
# 训练一个 epoch
# =====================================================
def train_epoch(net_LFN, net_LAN, train_loader, optimizer,
                criterion_l1, perceptual_loss, gradient_loss, epoch, args):
    net_LFN.train()
    net_LAN.train()

    epoch_loss = 0.0
    train_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}")

    for low_img, high_img in train_tqdm:
        low_img = low_img.cuda()
        high_img = high_img.cuda()

        optimizer.zero_grad()

        # 前向传播
        gain = net_LFN(low_img)
        _, enhanced_img, _ = net_LAN(low_img, gain)

        # 计算所有损失
        loss_l1 = criterion_l1(enhanced_img, high_img)
        loss_ssim = 1 - ssim(enhanced_img, high_img)
        loss_perceptual = perceptual_loss(enhanced_img, high_img)
        loss_perceptual = torch.sqrt(loss_perceptual + 1e-6)
        loss_gradient = gradient_loss(enhanced_img, high_img)

        # 加权组合
        total_loss = (
                1.0 * loss_l1 +
                0.4 * loss_ssim +
                0.3 * loss_perceptual +
                0.2 * loss_gradient
        )

        # 反向传播
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        train_tqdm.set_postfix(
            L1=f"{loss_l1.item():.3f}",
            SSIM=f"{loss_ssim.item():.3f}",
            Perc=f"{loss_perceptual.item():.3f}",
            Grad=f"{loss_gradient.item():.3f}",
            Total=f"{total_loss.item():.3f}"
        )

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss


# =====================================================
# 验证
# =====================================================
def validate(net_LFN, net_LAN, val_loader, criterion_l1, perceptual_loss, gradient_loss):
    net_LFN.eval()
    net_LAN.eval()

    val_loss = 0.0

    with torch.no_grad():
        for low_img, high_img in val_loader:
            low_img = low_img.cuda()
            high_img = high_img.cuda()

            gain = net_LFN(low_img)
            _, enhanced_img, _ = net_LAN(low_img, gain)

            loss_l1 = criterion_l1(enhanced_img, high_img)
            loss_ssim = 1 - ssim(enhanced_img, high_img)
            loss_perceptual = perceptual_loss(enhanced_img, high_img)
            loss_perceptual = torch.sqrt(loss_perceptual + 1e-6)
            loss_gradient = gradient_loss(enhanced_img, high_img)

            total_loss = (
                    1.0 * loss_l1 +
                    0.4 * loss_ssim +
                    0.3 * loss_perceptual +
                    0.2 * loss_gradient
            )

            val_loss += total_loss.item()

    avg_loss = val_loss / len(val_loader)
    return avg_loss


# =====================================================
# 训练一个配置
# =====================================================
def train_one_config(config, args, train_loader, val_loader, net_LFN,
                     criterion_l1, perceptual_loss, gradient_loss):
    """训练单个消融配置"""

    print(f"\n{'='*80}")
    print(f"🎯 训练配置: {config['name']}")
    print(f"{'='*80}")
    print(f"  - 使用 SA: {'✓' if config['use_sa'] else '✗'}")
    print(f"  - 使用 CA: {'✓' if config['use_ca'] else '✗'}")
    print(f"  - Spatial Weight: {args.spatial_weight}")
    print(f"{'='*80}\n")

    # ⭐ 创建 LAN 模型（根据配置）
    net_LAN = Spatial_Aware_Refinement(
        spatial_weight=args.spatial_weight,
        use_sa=config['use_sa'],
        use_ca=config['use_ca']
    ).cuda()

    # 加载 LAN 预训练权重（如果有）
    if args.lan_pretrain and os.path.exists(args.lan_pretrain):
        try:
            net_LAN.load_state_dict(torch.load(args.lan_pretrain), strict=False)
            print(f"✅ 加载 LAN 预训练权重: {args.lan_pretrain}")
        except Exception as e:
            print(f"⚠️  加载 LAN 预训练权重失败: {e}")
            print("   将从头开始训练")

    # 优化器（只优化 LAN，LFN 冻结）
    optimizer = optim.Adam(
        net_LAN.parameters(),  # ⭐ 只优化 LAN
        lr=args.lr,
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # 创建保存目录
    save_dir = os.path.join(args.save_path, config['save_name'])
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 模型保存目录: {save_dir}\n")

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_epoch(
            net_LFN, net_LAN, train_loader, optimizer,
            criterion_l1, perceptual_loss, gradient_loss, epoch, args
        )

        # 验证
        val_loss = validate(net_LFN, net_LAN, val_loader,
                            criterion_l1, perceptual_loss, gradient_loss)

        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch [{epoch}/{args.epochs}] - "
              f"Train: {train_loss:.4f}, "
              f"Val: {val_loss:.4f}, "
              f"LR: {current_lr:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'LAN_best.pth')
            torch.save(net_LAN.state_dict(), save_path)
            print(f"✅ 保存最佳模型 (Val Loss: {val_loss:.4f}) -> {save_path}")

        # 定期保存检查点
        if epoch % 20 == 0:
            save_path = os.path.join(save_dir, f'LAN_epoch_{epoch}.pth')
            torch.save(net_LAN.state_dict(), save_path)
            print(f"💾 保存检查点: epoch_{epoch} -> {save_path}")

    print(f"\n✓ 配置 {config['name']} 训练完成！最佳验证损失: {best_val_loss:.4f}\n")

    return best_val_loss


# =====================================================
# 初始化
# =====================================================
def init_seeds(seed=42):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# =====================================================
# 主函数
# =====================================================
def main(args):
    init_seeds(args.seed)

    print(f"\n{'='*80}")
    print(f"🚀 SA/CA 消融实验训练（基于您的原有脚本）")
    print(f"{'='*80}")
    print(f"🖥️  设备: CUDA")
    print(f"📊 训练配置:")
    print(f"  - 数据集: {args.dataset_path}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Spatial Weight: {args.spatial_weight}")
    print(f"\n🎯 损失函数:")
    print(f"  1. L1 Loss (权重 1.0)")
    print(f"  2. SSIM Loss (权重 0.4)")
    print(f"  3. Perceptual Loss (权重 0.3 + 平方根稳定化)")
    print(f"  4. Gradient Loss (权重 0.2)")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 加载数据
    dataset = PairwiseDataset(args.dataset_path, transform=transform)
    train_nums = int(len(dataset) * 0.9)
    val_nums = len(dataset) - train_nums
    train_data, val_data = torch.utils.data.random_split(dataset, [train_nums, val_nums])

    print(f"\n📊 数据划分: 训练 {train_nums} 张, 验证 {val_nums} 张")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =====================================================
    # 加载 LFN（冻结，不训练）
    # =====================================================
    print(f"\n📦 初始化 LFN（冻结）...")
    net_LFN = Global_Luminance_Perception().cuda()

    if args.lfn_pretrain and os.path.exists(args.lfn_pretrain):
        net_LFN.load_state_dict(torch.load(args.lfn_pretrain))
        print(f"✅ 加载 LFN 预训练: {args.lfn_pretrain}")
    else:
        print(f"⚠️  未找到 LFN 预训练权重，将使用随机初始化")

    net_LFN.eval()  # ⭐ 设置为评估模式
    for param in net_LFN.parameters():
        param.requires_grad = False  # ⭐ 冻结参数
    print("🔒 LFN 已冻结（不参与训练）")

    # =====================================================
    # 初始化损失函数
    # =====================================================
    print(f"\n🎯 初始化损失函数...")
    criterion_l1 = nn.L1Loss()
    perceptual_loss = PerceptualLoss().cuda()
    gradient_loss = GradientLoss().cuda()
    print("✅ 损失函数初始化完成")

    # =====================================================
    # 定义消融配置
    # =====================================================
    ablation_configs = [
        {
            'name': 'Full (SA+CA)',
            'use_sa': True,
            'use_ca': True,
            'save_name': 'Full_SA_CA'
        },
        {
            'name': 'w/o SA',
            'use_sa': False,
            'use_ca': True,
            'save_name': 'No_SA'
        },
        {
            'name': 'w/o CA',
            'use_sa': True,
            'use_ca': False,
            'save_name': 'No_CA'
        },
        {
            'name': 'w/o SA & CA',
            'use_sa': False,
            'use_ca': False,
            'save_name': 'No_SA_No_CA'
        },
    ]

    # =====================================================
    # 训练每个配置
    # =====================================================
    results = {}

    print(f"\n{'='*80}")
    print("🏋️  开始消融实验训练...")
    print(f"{'='*80}\n")

    for config in ablation_configs:
        best_loss = train_one_config(
            config, args, train_loader, val_loader, net_LFN,
            criterion_l1, perceptual_loss, gradient_loss
        )
        results[config['name']] = best_loss

    # =====================================================
    # 输出总结
    # =====================================================
    print(f"\n{'='*80}")
    print(f"🎉 所有消融实验训练完成！")
    print(f"{'='*80}")
    print(f"\n📊 各配置最佳验证损失:")
    print(f"{'-'*80}")

    for config_name, loss in results.items():
        print(f"  {config_name:20s} : {loss:.4f}")

    print(f"{'-'*80}")
    print(f"\n📁 所有模型保存在: {args.save_path}")
    print(f"{'='*80}\n")


# =====================================================
# 参数解析
# =====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SA/CA 消融实验训练')

    # 数据路径
    parser.add_argument('--dataset_path', default='./train/LFN_traingdata/our485',
                        help='数据集路径')
    parser.add_argument('--save_path', default='./checkpoint/ablation_checkpoints/',
                        help='模型保存路径')

    # 训练参数
    parser.add_argument('--epochs', default=40, type=int,
                        help='训练轮数')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='批次大小')
    parser.add_argument('--lr', default=0.0001, type=float,
                        help='学习率')

    # 模型参数
    parser.add_argument('--spatial_weight', default=0.5, type=float,
                        help='LAN 空间注意力权重')

    # 预训练权重
    parser.add_argument('--lfn_pretrain', default='./checkpoint/enhanced_train/LFN_epoch_20.pth',
                        type=str, help='LFN 预训练权重路径（必需，会被冻结）')
    parser.add_argument('--lan_pretrain', default='',
                        type=str, help='LAN 预训练权重路径（可选）')

    # 其他
    parser.add_argument('--seed', default=42, type=int,
                        help='随机种子')

    args = parser.parse_args()

    # 检查 LFN 预训练权重
    if not args.lfn_pretrain or not os.path.exists(args.lfn_pretrain):
        print(f"\n⚠️  警告: 未找到 LFN 预训练权重")
        print(f"   建议提供 --lfn_pretrain 参数")
        response = input("是否继续训练? (y/n): ")
        if response.lower() != 'y':
            print("训练已取消")
            exit(0)

    main(args)