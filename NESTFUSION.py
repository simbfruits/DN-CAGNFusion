import os
import re
import sys
import argparse
from PIL import Image

import torch
from torchvision import transforms

from DNFN.models import fuse_model, fusion_layer

# 添加 NestFuse 路径并导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'NestFuse'))

try:
    from DNFN.models.NestFuse import NestFuse_eval as DNFN_Model
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
# 颜色空间转换函数
# =====================================================
def rgb_to_ycbcr(input_im):
    """
    RGB to YCbCr 转换
    输入：[B, 3, H, W] 范围 [0, 1]
    输出：[B, 3, H, W] 范围 [0, 1]
    """
    device = input_im.device

    transform_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312]
    ], dtype=torch.float32, device=device).T

    img = input_im.permute(0, 2, 3, 1)
    ycbcr = torch.matmul(img, transform_matrix)
    ycbcr[:, :, :, 1:] = ycbcr[:, :, :, 1:] + 0.5
    ycbcr = torch.clamp(ycbcr, 0., 1.)
    return ycbcr.permute(0, 3, 1, 2)


def ycbcr2rgb(input_im):
    """
    YCbCr to RGB 转换
    输入：[B, 3, H, W] 范围 [0, 1]
    输出：[B, 3, H, W] 范围 [0, 1]
    """
    device = input_im.device

    transform_matrix = torch.tensor([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ], dtype=torch.float32, device=device).T

    img = input_im.permute(0, 2, 3, 1).clone()
    img[:, :, :, 1:] = img[:, :, :, 1:] - 0.5
    rgb = torch.matmul(img, transform_matrix)
    rgb = torch.clamp(rgb, 0., 1.)
    return rgb.permute(0, 3, 1, 2)


# =====================================================
# 保存张量为图像
# =====================================================
def save_tensor_as_image(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    to_pil = transforms.ToPILImage()
    img = to_pil(tensor)
    img.save(path)


def tensor_to_pil(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


# =====================================================
# 融合函数（仅 NestFuse）
# =====================================================
# def fusion_forward(vis, ir, DNFN, device, deepsupervision=True, debug=False):
def fusion_forward(vis, ir, DNFN, device, weight=0.5, deepsupervision=True, debug=False):
    """
    仅使用 NestFuse 的融合流程
    Args:
        vis: 可见光图像 tensor [1, 3, H, W]
        ir:  红外图像 tensor [1, 1, H, W]
        DNFN: NestFuse 模型
        device: 设备
        deepsupervision: 是否使用深度监督
        debug: 是否打印调试信息
    Returns:
        融合后的 RGB 图像 tensor
    """
    with torch.no_grad():
        if debug:
            print(f"\n{'=' * 50}")
            print(f"开始融合处理（仅 NestFuse）")
            print(f"{'=' * 50}")
            print(f"输入 vis shape: {vis.shape}, range: [{vis.min():.3f}, {vis.max():.3f}]")
            print(f"输入 ir  shape: {ir.shape},  range: [{ir.min():.3f}, {ir.max():.3f}]")

        # 1. 提取可见光的 Y / Cb / Cr 通道
        vis_ycbcr = rgb_to_ycbcr(vis)
        Y_vis   = vis_ycbcr[:, 0:1, :, :]
        Cb_orig = vis_ycbcr[:, 1:2, :, :]
        Cr_orig = vis_ycbcr[:, 2:3, :, :]

        if debug:
            print(f"\n色彩空间转换:")
            print(f"  Y_vis  range: [{Y_vis.min():.3f},   {Y_vis.max():.3f}]")
            print(f"  Cb     range: [{Cb_orig.min():.3f}, {Cb_orig.max():.3f}]")
            print(f"  Cr     range: [{Cr_orig.min():.3f}, {Cr_orig.max():.3f}]")
            print(f"  IR     range: [{ir.min():.3f},      {ir.max():.3f}]")

        # 2. Y_vis 和 IR 分别通过 DNFN encoder
        Y_vis_features = DNFN.encoder(Y_vis)
        ir_features    = DNFN.encoder(ir)

        # 3. 特征融合
        # fused_features = fusion_layer(Y_vis_features, ir_features)
        fused_features = fusion_layer(Y_vis_features, ir_features, weight=weight)
        # 4. 解码得到融合的亮度通道
        if deepsupervision:
            fused_outputs = DNFN.decoder(fused_features)
            F_gray = fused_outputs[2]  # 取最后一层输出
        else:
            F_gray = DNFN.decoder(fused_features)

        F_gray = torch.clamp(F_gray, 0., 1.)

        if debug:
            print(f"\nDNFN 融合输出:")
            print(f"  F_gray shape: {F_gray.shape}, range: [{F_gray.min():.3f}, {F_gray.max():.3f}]")

        # 5. 融合亮度 + 原始色度 -> YCbCr -> RGB
        fusion_ycbcr = torch.cat((F_gray, Cb_orig, Cr_orig), dim=1)
        F_rgb = ycbcr2rgb(fusion_ycbcr)
        F_rgb = torch.clamp(F_rgb, 0., 1.)

        if debug:
            print(f"\n最终输出:")
            print(f"  F_rgb range: [{F_rgb.min():.3f}, {F_rgb.max():.3f}]")
            print(f"  R: [{F_rgb[:, 0].min():.3f}, {F_rgb[:, 0].max():.3f}]")
            print(f"  G: [{F_rgb[:, 1].min():.3f}, {F_rgb[:, 1].max():.3f}]")
            print(f"  B: [{F_rgb[:, 2].min():.3f}, {F_rgb[:, 2].max():.3f}]")
            print(f"{'=' * 50}\n")

        return F_rgb



# =====================================================
# 主函数
# =====================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()

    print(f"使用设备: {device}")

    # 加载 DNFN 模型
    print('正在加载 NestFuse 模型...')
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
    IR_path  = args.ir_dir
    VIS_path = args.vis_dir

    if not os.path.exists(IR_path):
        print(f"❌ 错误: 红外图像目录不存在: {IR_path}")
        return
    if not os.path.exists(VIS_path):
        print(f"❌ 错误: 可见光图像目录不存在: {VIS_path}")
        return

    IR_image_list  = os.listdir(IR_path)
    VIS_image_list = os.listdir(VIS_path)

    # 按数字排序
    IR_image_list  = sorted(IR_image_list,  key=lambda i: int(re.search(r'(\d+)', i).group()))
    VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))

    print(f'找到 {len(IR_image_list)} 对图像')
    print('开始融合...')

    num = 0
    for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
        num += 1
        IR_image_path  = os.path.join(IR_path,  IR_image_name)
        VIS_image_path = os.path.join(VIS_path, VIS_image_name)

        # 读取图像
        img_vis = Image.open(VIS_image_path).convert('RGB')
        img_ir  = Image.open(IR_image_path).convert('L')

        # 转换为 tensor
        vis = to_tensor(img_vis).unsqueeze(0).to(device)
        ir  = to_tensor(img_ir).unsqueeze(0).to(device)

        # 融合（只在第一张图像时打印调试信息）
        debug_mode = (num == 1) and args.debug
        fused = fusion_forward(vis, ir, DNFN, device,
                               weight=args.fusion_weight,
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
    parser = argparse.ArgumentParser(description='图像融合测试（仅 NestFuse）')

    # 数据路径
    parser.add_argument('--vis_dir', type=str, default='./DNFN/fusion_test_data/L/vi',
                        help='可见光图像目录')
    parser.add_argument('--ir_dir', type=str, default='./DNFN/fusion_test_data/L/ir',
                        help='红外图像目录')
    parser.add_argument('--result_path', type=str, default='./DNFN/fusion_test_data/L/DNFN',
                        help='融合结果保存目录')

    # 模型路径
    parser.add_argument('--dnfn_path', type=str,
                        default='./DNFN/runs/train_12-04_16-01/checkpoints/epoch018-loss0.338.pth',
                        help='NestFuse 模型权重路径')

    # 模型参数
    parser.add_argument('--deepsupervision', type=bool, default=True,
                        help='DNFN 是否使用深度监督')
    parser.add_argument('--debug', action='store_true',
                        help='是否打印详细调试信息')
    parser.add_argument('--fusion_weight', type=float, default=0.5,
                        help='红外融合权重比例 (0:完全可见光, 1:完全红外)')

    args = parser.parse_args()
    main(args)