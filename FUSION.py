import os
import re
import sys
import argparse
from PIL import Image

import torch
from torchvision import transforms

from model_GLPN import Global_Luminance_Perception
from model_SARN import Spatial_Aware_Refinement
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
    device = input_im.device
    transform_matrix = torch.tensor([
        [0.299,    0.587,    0.114  ],
        [-0.168736,-0.331264, 0.5   ],
        [0.5,     -0.418688,-0.081312]
    ], dtype=torch.float32, device=device).T
    img   = input_im.permute(0, 2, 3, 1)
    ycbcr = torch.matmul(img, transform_matrix)
    ycbcr[:, :, :, 1:] = ycbcr[:, :, :, 1:] + 0.5
    ycbcr = torch.clamp(ycbcr, 0., 1.)
    return ycbcr.permute(0, 3, 1, 2)


def ycbcr2rgb(input_im):
    device = input_im.device
    transform_matrix = torch.tensor([
        [1.0,  0.0,       1.402  ],
        [1.0, -0.344136, -0.714136],
        [1.0,  1.772,     0.0    ]
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
    transforms.ToPILImage()(tensor).save(path)


def tensor_to_pil(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)


# =====================================================
# 融合函数
# =====================================================
def fusion_forward(vis, ir, GLP, SAR, DNFN, device, weight,
                   deepsupervision=True, debug=False,
                   save_enhanced=False, enhanced_path=None):
    """
    完整的融合流程
    Args:
        vis            : 可见光图像 tensor [1, 3, H, W]
        ir             : 红外图像 tensor [1, 1, H, W]
        GLP            : LFN 模型
        SAR            : SARN 模型
        DNFN           : DNFN 模型
        device         : 设备
        weight         : 红外融合权重
        deepsupervision: 是否使用深度监督
        debug          : 是否打印调试信息
        save_enhanced  : ⭐ 是否保存 GLPN+SARN 增强后的彩色图像
        enhanced_path  : ⭐ 增强图像保存路径（save_enhanced=True 时必须提供）
    Returns:
        融合后的 RGB 图像 tensor
    """
    print(f"DEBUG: 正在使用权重 weight = {weight}")
    with torch.no_grad():
        if debug:
            print(f"\n{'=' * 50}")
            print(f"开始融合处理")
            print(f"{'=' * 50}")
            print(f"输入 vis shape: {vis.shape}, range: [{vis.min():.3f}, {vis.max():.3f}]")
            print(f"输入 ir  shape: {ir.shape},  range: [{ir.min():.3f}, {ir.max():.3f}]")

        # --------------------------------------------------
        # 1. LFN 预测增益 + SARN 增强可见光图像
        # --------------------------------------------------
        gain = GLP(vis)
        if debug:
            print(f"\nLFN 输出 gain: {gain.shape}, values: {gain.cpu().numpy()}")

        _, A_en, _ = SAR(vis, gain)
        A_en = torch.clamp(A_en, 0., 1.)

        if debug:
            print(f"SARN 增强后 A_en: shape {A_en.shape}, "
                  f"range: [{A_en.min():.3f}, {A_en.max():.3f}]")

        # --------------------------------------------------
        # ⭐ 可选：将增强后的彩色图像保存到磁盘
        # --------------------------------------------------
        if save_enhanced and enhanced_path is not None:
            save_tensor_as_image(A_en, enhanced_path)
            if debug:
                print(f"  [保存增强图] {enhanced_path}")

        # --------------------------------------------------
        # 2. 提取原始可见光的色度通道
        # --------------------------------------------------
        vis_ycbcr = rgb_to_ycbcr(vis)
        Cb_orig   = vis_ycbcr[:, 1:2, :, :]
        Cr_orig   = vis_ycbcr[:, 2:3, :, :]

        # 提取增强后的亮度通道
        A_en_ycbcr = rgb_to_ycbcr(A_en)
        Y_en       = A_en_ycbcr[:, 0:1, :, :]

        if debug:
            print(f"\n色彩空间转换:")
            print(f"  Y_en   range: [{Y_en.min():.3f},    {Y_en.max():.3f}]")
            print(f"  Cb     range: [{Cb_orig.min():.3f}, {Cb_orig.max():.3f}]")
            print(f"  Cr     range: [{Cr_orig.min():.3f}, {Cr_orig.max():.3f}]")
            print(f"  IR     range: [{ir.min():.3f},      {ir.max():.3f}]")

        # --------------------------------------------------
        # 3. Y_en 和 IR 分别通过 DNFN encoder
        # --------------------------------------------------
        Y_en_features = DNFN.encoder(Y_en)
        ir_features   = DNFN.encoder(ir)

        # --------------------------------------------------
        # 4. 特征融合
        # --------------------------------------------------
        fused_features = fusion_layer(Y_en_features, ir_features, weight=weight)

        # --------------------------------------------------
        # 5. 解码得到融合的亮度通道
        # --------------------------------------------------
        if deepsupervision:
            fused_outputs = DNFN.decoder(fused_features)
            F_gray = fused_outputs[2]
        else:
            F_gray = DNFN.decoder(fused_features)

        F_gray = torch.clamp(F_gray, 0., 1.)

        if debug:
            print(f"\nDNFN 融合输出:")
            print(f"  F_gray shape: {F_gray.shape}, "
                  f"range: [{F_gray.min():.3f}, {F_gray.max():.3f}]")

        # --------------------------------------------------
        # 6. YCbCr -> RGB（融合亮度 + 原始色度）
        # --------------------------------------------------
        fusion_ycbcr = torch.cat((F_gray, Cb_orig, Cr_orig), dim=1)

        if debug:
            print(f"\n准备转换回 RGB:")
            print(f"  Y  range: [{fusion_ycbcr[:,0:1].min():.3f}, {fusion_ycbcr[:,0:1].max():.3f}]")
            print(f"  Cb range: [{fusion_ycbcr[:,1:2].min():.3f}, {fusion_ycbcr[:,1:2].max():.3f}]")
            print(f"  Cr range: [{fusion_ycbcr[:,2:3].min():.3f}, {fusion_ycbcr[:,2:3].max():.3f}]")

        F_rgb = ycbcr2rgb(fusion_ycbcr)
        F_rgb = torch.clamp(F_rgb, 0., 1.)

        if debug:
            print(f"\n最终输出:")
            print(f"  F_rgb range: [{F_rgb.min():.3f}, {F_rgb.max():.3f}]")
            print(f"  R: [{F_rgb[:,0].min():.3f}, {F_rgb[:,0].max():.3f}]")
            print(f"  G: [{F_rgb[:,1].min():.3f}, {F_rgb[:,1].max():.3f}]")
            print(f"  B: [{F_rgb[:,2].min():.3f}, {F_rgb[:,2].max():.3f}]")
            print(f"{'=' * 50}\n")

        return F_rgb


# =====================================================
# 主函数
# =====================================================
def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = transforms.ToTensor()
    print(f"使用设备: {device}")

    # ── 加载 LFN 和 SARN 模型 ────────────────────────────
    print('正在加载 LFN 和 SARN 模型...')
    GLP = Global_Luminance_Perception().to(device).eval()
    SAR = Spatial_Aware_Refinement(spatial_weight=args.spatial_weight).to(device).eval()

    if not os.path.exists(args.lfn_path):
        print(f"❌ LFN 权重文件不存在: {args.lfn_path}"); return
    if not os.path.exists(args.lan_path):
        print(f"❌ SARN 权重文件不存在: {args.lan_path}"); return

    GLP.load_state_dict(torch.load(args.lfn_path, map_location=device))
    SAR.load_state_dict(torch.load(args.lan_path, map_location=device))
    print(f'✅ LFN 和 SARN 加载完成 (spatial_weight={args.spatial_weight})')

    # ── 加载 DNFN 模型 ───────────────────────────────────
    print('正在加载 DNFN 模型...')
    DNFN = DNFN_Model(input_nc=1, output_nc=1).to(device).eval()

    if not os.path.exists(args.dnfn_path):
        print(f"❌ DNFN 权重文件不存在: {args.dnfn_path}"); return

    checkpoint = torch.load(args.dnfn_path, map_location=device)
    if 'encoder_state_dict' in checkpoint:
        DNFN.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        DNFN.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    elif 'model' in checkpoint:
        DNFN.load_state_dict(checkpoint['model'])
    else:
        DNFN.load_state_dict(checkpoint)
    print('✅ DNFN 加载完成')

    # ── 创建输出目录 ──────────────────────────────────────
    os.makedirs(args.result_path, exist_ok=True)

    # ⭐ 创建增强图像保存目录（仅当开关打开时）
    if args.save_enhanced:
        os.makedirs(args.enhanced_path, exist_ok=True)
        print(f'✅ 增强图像将保存至: {args.enhanced_path}')

    # ── 载入图像列表 ──────────────────────────────────────
    print('载入数据...')
    if not os.path.exists(args.ir_dir):
        print(f"❌ 红外图像目录不存在: {args.ir_dir}"); return
    if not os.path.exists(args.vis_dir):
        print(f"❌ 可见光图像目录不存在: {args.vis_dir}"); return

    IR_image_list  = sorted(os.listdir(args.ir_dir),
                            key=lambda i: int(re.search(r'(\d+)', i).group()))
    VIS_image_list = sorted(os.listdir(args.vis_dir),
                            key=lambda i: int(re.search(r'(\d+)', i).group()))

    print(f'找到 {len(IR_image_list)} 对图像')
    print('开始融合...')

    num = 0
    for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
        num += 1
        IR_image_path  = os.path.join(args.ir_dir,  IR_image_name)
        VIS_image_path = os.path.join(args.vis_dir, VIS_image_name)

        img_vis = Image.open(VIS_image_path).convert('RGB')
        img_ir  = Image.open(IR_image_path).convert('L')

        vis = to_tensor(img_vis).unsqueeze(0).to(device)
        ir  = to_tensor(img_ir).unsqueeze(0).to(device)

        # ⭐ 增强图像保存路径（文件名与融合结果保持一致）
        enhanced_save_path = (
            os.path.join(args.enhanced_path, f'{num}.png')
            if args.save_enhanced else None
        )

        debug_mode = (num == 1) and args.debug
        fused = fusion_forward(
            vis, ir, GLP, SAR, DNFN, device,
            weight          = args.fusion_weight,
            deepsupervision = args.deepsupervision,
            debug           = debug_mode,
            save_enhanced   = args.save_enhanced,       # ⭐ 开关
            enhanced_path   = enhanced_save_path,       # ⭐ 路径
        )

        # 保存融合结果
        tensor_to_pil(fused).save(os.path.join(args.result_path, f'{num}.png'))

        if num % 10 == 0 or num == len(IR_image_list):
            print(f'[{num}/{len(IR_image_list)}] 已完成')

    print('✅ 融合完成！')
    print(f'融合结果保存在: {args.result_path}')
    if args.save_enhanced:
        print(f'增强结果保存在: {args.enhanced_path}')


# =====================================================
# 参数入口
# =====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='图像融合测试')

    # ── 数据路径 ──────────────────────────────────────────
    parser.add_argument('--vis_dir',     type=str, default='./DNFN/fusion_test_data/L/vi')
    parser.add_argument('--ir_dir',      type=str, default='./DNFN/fusion_test_data/L/ir')
    parser.add_argument('--result_path', type=str, default='./DNFN/fusion_test_data/L/CAGN')

    # ── 模型路径 ──────────────────────────────────────────
    parser.add_argument('--lfn_path',  type=str,
                        default='./checkpoint/enhanced_train/LFN_epoch_20.pth')
    parser.add_argument('--lan_path',  type=str,
                        default='./checkpoint/enhanced_train/LAN_epoch_20.pth')
    parser.add_argument('--dnfn_path', type=str,
                        default='./DNFN/runs/train_12-04_16-01/checkpoints/epoch018-loss0.338.pth')

    # ── 模型参数 ──────────────────────────────────────────
    parser.add_argument('--deepsupervision', type=bool,  default=True)
    parser.add_argument('--spatial_weight',  type=float, default=0.5)
    parser.add_argument('--fusion_weight',   type=float, default=0.5,
                        help='红外融合权重比例 (0:完全可见光, 1:完全红外)')
    parser.add_argument('--debug', action='store_true')

    # ── ⭐ 增强图像输出开关 ───────────────────────────────
    parser.add_argument(
        '--save_enhanced',
        action='store_true',           # 命令行加 --save_enhanced 即开启，不加则关闭
        help='是否保存 GLPN+SARN 增强后的彩色图像'
    )
    parser.add_argument(
        '--enhanced_path',
        type=str,
        default=r'E:\pycharm\Project\DN-CAGNFusion\DN-CAGNFusion\DNFN\fusion_test_data\L\EN',
        help='增强图像保存目录（--save_enhanced 开启时生效）'
    )

    args = parser.parse_args()
    main(args)