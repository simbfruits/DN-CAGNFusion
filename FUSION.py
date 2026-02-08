import os
import re
import sys
import argparse
from PIL import Image

import torch
from torchvision import transforms

from model_GLPN import Global_Luminance_Perception
from model_SARN import Spatial_Aware_Refinement
from NestFuse.models import fuse_model, fusion_layer

# 添加 NestFuse 路径并导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NestFuse'))

try:
    from NestFuse.models.NestFuse import NestFuse_eval as DNFN_Model

    print("✅ DNFN 模型导入成功")
except ImportError as e:
    print(f"⚠️ 导入失败，尝试备用路径: {e}")
    try:
        from models.NestFuse import NestFuse_eval as DNFN_Model

        print("✅ DNFN 模型导入成功（备用路径）")
    except ImportError as e2:
        print(f"❌ 导入失败: {e2}")
        raise


# =====================================================
# 颜色空间转换函数（修复版 - 方案1）
# =====================================================
def rgb_to_ycbcr(input_im):
    """
    RGB to YCbCr 转换（修复版）
    输入：[B, 3, H, W] 范围 [0, 1]
    输出：[B, 3, H, W] 范围 [0, 1]

    使用标准的 ITU-R BT.601 转换矩阵
    """
    device = input_im.device

    # 标准 RGB to YCbCr 转换矩阵
    # Y  = 0.299*R + 0.587*G + 0.114*B
    # Cb = -0.168736*R - 0.331264*G + 0.5*B + 0.5
    # Cr = 0.5*R - 0.418688*G - 0.081312*B + 0.5
    transform_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], dtype=torch.float32, device=device).T  # 转置为 [3, 3]

    # [B, 3, H, W] -> [B, H, W, 3]
    img = input_im.permute(0, 2, 3, 1)

    # 应用转换矩阵
    ycbcr = torch.matmul(img, transform_matrix)

    # Cb 和 Cr 通道添加偏移量 0.5，归一化到 [0, 1]
    ycbcr[:, :, :, 1:] = ycbcr[:, :, :, 1:] + 0.5

    # 限制范围到 [0, 1]
    ycbcr = torch.clamp(ycbcr, 0., 1.)

    # [B, H, W, 3] -> [B, 3, H, W]
    return ycbcr.permute(0, 3, 1, 2)


def ycbcr2rgb(input_im):
    """
    YCbCr to RGB 转换（修复版）
    输入：[B, 3, H, W] 范围 [0, 1]
    输出：[B, 3, H, W] 范围 [0, 1]

    使用标准的 ITU-R BT.601 逆转换矩阵
    """
    device = input_im.device

    # 标准 YCbCr to RGB 逆转换矩阵
    # R = Y + 1.402*(Cr-0.5)
    # G = Y - 0.344136*(Cb-0.5) - 0.714136*(Cr-0.5)
    # B = Y + 1.772*(Cb-0.5)
    transform_matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], dtype=torch.float32, device=device).T  # 转置为 [3, 3]

    # [B, 3, H, W] -> [B, H, W, 3]
    img = input_im.permute(0, 2, 3, 1).clone()

    # 从 Cb 和 Cr 通道中减去偏移量 0.5
    img[:, :, :, 1:] = img[:, :, :, 1:] - 0.5

    # 应用逆转换矩阵
    rgb = torch.matmul(img, transform_matrix)

    # 限制范围到 [0, 1]
    rgb = torch.clamp(rgb, 0., 1.)

    # [B, H, W, 3] -> [B, 3, H, W]
    return rgb.permute(0, 3, 1, 2)


# =====================================================
# 特征融合函数
# =====================================================
# def fusion_layer(image1_EN, image2_EN):
#     """
#     特征融合策略 (L1-norm 加权)
#     """
#     if isinstance(image1_EN, (list, tuple)):
#         Fusion_features = []
#         for feat1, feat2 in zip(image1_EN, image2_EN):
#             w1 = torch.abs(feat1)
#             w2 = torch.abs(feat2)
#             w_sum = w1 + w2 + 1e-10
#             fused = (w1 * feat1 + w2 * feat2) / w_sum
#             Fusion_features.append(fused)
#         return Fusion_features
#     else:
#         w1 = torch.abs(image1_EN)
#         w2 = torch.abs(image2_EN)
#         w_sum = w1 + w2 + 1e-10
#         Fusion_feature = (w1 * image1_EN + w2 * image2_EN) / w_sum
#         return Fusion_feature


# =====================================================
# 保存张量为图像
# =====================================================
def save_tensor_as_image(tensor, path):
    """保存张量为图像文件"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(path)


def tensor_to_pil(tensor):
    """张量转 PIL 图像"""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


# =====================================================
# 融合函数（修复版）
# =====================================================
def fusion_forward(vis, ir, GLP, SAR, DNFN, device, deepsupervision=True, debug=False):
    """
    完整的融合流程（修复版）
    Args:
        vis: 可见光图像 tensor [1, 3, H, W]
        ir: 红外图像 tensor [1, 1, H, W]
        GLP: LFN 模型
        SAR: SAR 模型
        DNFN: DNFN 模型
        device: 设备
        deepsupervision: 是否使用深度监督
        debug: 是否打印调试信息
    Returns:
        融合后的 RGB 图像 tensor
    """
    with torch.no_grad():
        if debug:
            print(f"\n{'=' * 50}")
            print(f"开始融合处理")
            print(f"{'=' * 50}")
            print(f"输入 vis shape: {vis.shape}, range: [{vis.min():.3f}, {vis.max():.3f}]")
            print(f"输入 ir shape: {ir.shape}, range: [{ir.min():.3f}, {ir.max():.3f}]")

        # 1. LFN 预测增益 + LAN 增强可见光图像
        gain = GLP(vis)
        if debug:
            print(f"\nLFN 输出 gain: {gain.shape}, values: {gain.cpu().numpy()}")

        _, A_en, _ = SAR(vis, gain)
        A_en = torch.clamp(A_en, 0., 1.)

        if debug:
            print(f"LAN 增强后 A_en: shape {A_en.shape}, range: [{A_en.min():.3f}, {A_en.max():.3f}]")

        # 2. 提取原始可见光的色度通道（关键修改）
        vis_ycbcr = rgb_to_ycbcr(vis)
        Cb_orig = vis_ycbcr[:, 1:2, :, :]
        Cr_orig = vis_ycbcr[:, 2:3, :, :]

        # 提取增强后的亮度通道
        A_en_ycbcr = rgb_to_ycbcr(A_en)
        Y_en = A_en_ycbcr[:, 0:1, :, :]

        if debug:
            print(f"\n色彩空间转换:")
            print(f"  Y_en (增强亮度) range: [{Y_en.min():.3f}, {Y_en.max():.3f}]")
            print(f"  Cb_orig (原始色度) range: [{Cb_orig.min():.3f}, {Cb_orig.max():.3f}]")
            print(f"  Cr_orig (原始色度) range: [{Cr_orig.min():.3f}, {Cr_orig.max():.3f}]")
            print(f"  IR range: [{ir.min():.3f}, {ir.max():.3f}]")

        # 3. Y_en 和 IR 分别通过 DNFN encoder
        Y_en_features = DNFN.encoder(Y_en)
        ir_features = DNFN.encoder(ir)

        # 4. 特征融合
        fused_features = fusion_layer(Y_en_features, ir_features)

        # 5. 解码得到融合的亮度通道
        if deepsupervision:
            fused_outputs = DNFN.decoder(fused_features)
            F_gray = fused_outputs[2]  # 取最后一层输出
        else:
            F_gray = DNFN.decoder(fused_features)

        F_gray = torch.clamp(F_gray, 0., 1.)

        if debug:
            print(f"\nDNFN 融合输出:")
            print(f"  F_gray shape: {F_gray.shape}, range: [{F_gray.min():.3f}, {F_gray.max():.3f}]")

        # 6. YCbCr -> RGB（融合的亮度 + 原始色度）
        fusion_ycbcr = torch.cat((F_gray, Cb_orig, Cr_orig), dim=1)

        if debug:
            print(f"\n准备转换回 RGB:")
            print(f"  fusion_ycbcr shape: {fusion_ycbcr.shape}")
            print(f"  Y channel range: [{fusion_ycbcr[:, 0:1].min():.3f}, {fusion_ycbcr[:, 0:1].max():.3f}]")
            print(f"  Cb channel range: [{fusion_ycbcr[:, 1:2].min():.3f}, {fusion_ycbcr[:, 1:2].max():.3f}]")
            print(f"  Cr channel range: [{fusion_ycbcr[:, 2:3].min():.3f}, {fusion_ycbcr[:, 2:3].max():.3f}]")

        F_rgb = ycbcr2rgb(fusion_ycbcr)
        F_rgb = torch.clamp(F_rgb, 0., 1.)

        if debug:
            print(f"\n最终输出:")
            print(f"  F_rgb range: [{F_rgb.min():.3f}, {F_rgb.max():.3f}]")
            print(f"  R channel: [{F_rgb[:, 0].min():.3f}, {F_rgb[:, 0].max():.3f}]")
            print(f"  G channel: [{F_rgb[:, 1].min():.3f}, {F_rgb[:, 1].max():.3f}]")
            print(f"  B channel: [{F_rgb[:, 2].min():.3f}, {F_rgb[:, 2].max():.3f}]")
            print(f"{'=' * 50}\n")

        return F_rgb


# =====================================================
# 主函数
# =====================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    print(f"使用设备: {device}")

    # 加载 LFN 和 LAN 模型
    print('正在加载 LFN 和 LAN 模型...')
    GLP = Global_Luminance_Perception().to(device).eval()
    SAR = Spatial_Aware_Refinement(spatial_weight=args.spatial_weight).to(device).eval()

    # 检查权重文件是否存在
    if not os.path.exists(args.lfn_path):
        print(f"❌ 错误: LFN 权重文件不存在: {args.lfn_path}")
        return
    if not os.path.exists(args.lan_path):
        print(f"❌ 错误: LAN 权重文件不存在: {args.lan_path}")
        return

    GLP.load_state_dict(torch.load(args.lfn_path, map_location=device))
    SAR.load_state_dict(torch.load(args.lan_path, map_location=device))
    print(f'✅ LFN 和 LAN 加载完成 (spatial_weight={args.spatial_weight})')

    # 加载 DNFN 模型
    print('正在加载 DNFN 模型...')
    DNFN = DNFN_Model(input_nc=1, output_nc=1).to(device).eval()

    if not os.path.exists(args.dnfn_path):
        print(f"❌ 错误: DNFN 权重文件不存在: {args.dnfn_path}")
        return

    checkpoint = torch.load(args.dnfn_path, map_location=device)
    if 'encoder_state_dict' in checkpoint:
        DNFN.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        DNFN.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    elif 'model' in checkpoint:
        DNFN.load_state_dict(checkpoint['model'])
    else:
        DNFN.load_state_dict(checkpoint)
    print('✅ DNFN 加载完成')

    # 创建保存目录
    os.makedirs(args.result_path, exist_ok=True)

    print('载入数据...')
    IR_path = args.ir_dir
    VIS_path = args.vis_dir

    if not os.path.exists(IR_path):
        print(f"❌ 错误: 红外图像目录不存在: {IR_path}")
        return
    if not os.path.exists(VIS_path):
        print(f"❌ 错误: 可见光图像目录不存在: {VIS_path}")
        return

    IR_image_list = os.listdir(IR_path)
    VIS_image_list = os.listdir(VIS_path)

    # 按数字排序
    IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))

    print(f'找到 {len(IR_image_list)} 对图像')
    print('开始融合...')

    num = 0
    for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
        num += 1
        IR_image_path = os.path.join(IR_path, IR_image_name)
        VIS_image_path = os.path.join(VIS_path, VIS_image_name)

        # 读取图像
        img_vis = Image.open(VIS_image_path).convert('RGB')
        img_ir = Image.open(IR_image_path).convert('L')

        # 转换为 tensor
        vis = to_tensor(img_vis).unsqueeze(0).to(device)
        ir = to_tensor(img_ir).unsqueeze(0).to(device)

        # 融合 (只在第一张图像时打印调试信息)
        debug_mode = (num == 1) and args.debug
        fused = fusion_forward(vis, ir, GLP, SAR, DNFN, device,
                               deepsupervision=args.deepsupervision,
                               debug=debug_mode)

        # 保存结果
        fused_img = tensor_to_pil(fused)
        save_path = os.path.join(args.result_path, f'{num}.png')
        fused_img.save(save_path)

        if num % 10 == 0 or num == len(IR_image_list):
            print(f'[{num}/{len(IR_image_list)}] 已完成')

    print('✅ 融合完成！')
    print(f'结果保存在: {args.result_path}')


# =====================================================
# 参数入口
# =====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像融合测试')

    # 数据路径
    parser.add_argument('--vis_dir', type=str, default='./NestFuse/fusion_test_data/msrs/vi',
                        help='可见光图像目录')
    parser.add_argument('--ir_dir', type=str, default='./NestFuse/fusion_test_data/msrs/ir',
                        help='红外图像目录')
    parser.add_argument('--result_path', type=str, default='./test_results/msrs_',
                        help='融合结果保存目录')

    # 模型路径
    # parser.add_argument('--lfn_path', type=str, default='./checkpoint/LFN_epoch_100.pth',
    #                     help='LFN 模型权重路径')
    # parser.add_argument('--lan_path', type=str, default='./checkpoint/LAN_epoch_100.pth',
    #                     help='LAN 模型权重路径')
    # parser.add_argument('--lfn_path', type=str, default='./checkpoint/enhanced_train/LFN_epoch_20.pth',
    #                     help='LFN 模型权重路径')
    # parser.add_argument('--lan_path', type=str, default='./checkpoint/enhanced_train/LAN_epoch_20.pth',
    #                     help='LAN 模型权重路径')
    parser.add_argument('--lfn_path', type=str, default='./checkpoint/enhanced_train/LFN_epoch_20.pth',
                        help='LFN 模型权重路径')
    parser.add_argument('--lan_path', type=str, default='./checkpoint/enhanced_train/LAN_epoch_20.pth',
                        help='LAN 模型权重路径')
    parser.add_argument('--dnfn_path', type=str,
                        default='./NestFuse/runs/train_12-04_16-01/checkpoints/epoch018-loss0.338.pth',
                        help='DNFN 模型权重路径')

    # 模型参数
    parser.add_argument('--deepsupervision', type=bool, default=True,
                        help='DNFN 是否使用深度监督')
    parser.add_argument('--spatial_weight', type=float, default=0.5,
                        help='LAN 空间注意力权重 (0-1), 越小作用越弱')
    parser.add_argument('--debug', action='store_true',
                        help='是否打印详细调试信息')

    args = parser.parse_args()
    main(args)