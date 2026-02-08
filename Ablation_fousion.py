"""
消融实验测试脚本 - 仅保存融合图片
基于您的 FUSION.py，测试不同的 SA/CA 配置，只输出融合结果图像
"""
import re
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# 导入模型
from model_GLPN import Global_Luminance_Perception
from model_SARN import Spatial_Aware_Refinement
from NestFuse.models.NestFuse import NestFuse_eval as DNFN_Model
from FUSION import rgb_to_ycbcr, ycbcr2rgb
from NestFuse.models import fusion_layer


def test_fusion(vis, ir, LFN, LAN, DNFN, device):
    """执行融合"""
    with torch.no_grad():
        # LFN 预测增益
        gain = LFN(vis)

        # LAN 增强
        _, A_en, _ = LAN(vis, gain)
        A_en = torch.clamp(A_en, 0., 1.)

        # 颜色空间转换
        vis_ycbcr = rgb_to_ycbcr(vis)
        Cb_orig = vis_ycbcr[:, 1:2, :, :]
        Cr_orig = vis_ycbcr[:, 2:3, :, :]

        A_en_ycbcr = rgb_to_ycbcr(A_en)
        Y_en = A_en_ycbcr[:, 0:1, :, :]

        # DNFN 特征提取和融合
        Y_en_features = DNFN.encoder(Y_en)
        ir_features = DNFN.encoder(ir)

        fused_features = fusion_layer(Y_en_features, ir_features)

        # 解码
        fused_outputs = DNFN.decoder(fused_features)
        F_gray = fused_outputs[2]
        F_gray = torch.clamp(F_gray, 0., 1.)

        # 颜色重建
        fusion_ycbcr = torch.cat((F_gray, Cb_orig, Cr_orig), dim=1)
        F_rgb = ycbcr2rgb(fusion_ycbcr)
        F_rgb = torch.clamp(F_rgb, 0., 1.)

        return F_rgb


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ========================================
    # 1. 加载固定模型 (LFN, DNFN)
    # ========================================
    print("\n[1/4] 加载固定模型...")

    # LFN
    GLP = Global_Luminance_Perception().to(device).eval()
    GLP.load_state_dict(torch.load(args.lfn_path, map_location=device))
    print("✓ LFN 加载完成")

    # DNFN
    DNFN = DNFN_Model(input_nc=1, output_nc=1).to(device).eval()
    checkpoint = torch.load(args.dnfn_path, map_location=device)
    if 'encoder_state_dict' in checkpoint:
        DNFN.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        DNFN.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        DNFN.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    print("✓ DNFN 加载完成")

    # ========================================
    # 2. 定义消融配置
    # ========================================
    ablation_configs = [
        {
            'name': 'Full (SA+CA)',
            'use_sa': True,
            'use_ca': True,
            'weight_path': os.path.join(args.checkpoint_dir, 'Full_SA_CA', 'LAN_best.pth')
        },
        {
            'name': 'w/o SA',
            'use_sa': False,
            'use_ca': True,
            'weight_path': os.path.join(args.checkpoint_dir, 'No_SA', 'LAN_best.pth')
        },
        {
            'name': 'w/o CA',
            'use_sa': True,
            'use_ca': False,
            'weight_path': os.path.join(args.checkpoint_dir, 'No_CA', 'LAN_best.pth')
        },
        {
            'name': 'w/o SA&CA',
            'use_sa': False,
            'use_ca': False,
            'weight_path': os.path.join(args.checkpoint_dir, 'No_SA_No_CA', 'LAN_best.pth')
        },
    ]

    # ========================================
    # 3. 准备测试数据
    # ========================================
    print("\n[2/4] 准备测试数据...")

    IR_path = args.ir_dir
    VIS_path = args.vis_dir

    # IR_image_list = sorted(os.listdir(IR_path))
    # VIS_image_list = sorted(os.listdir(VIS_path))
    IR_image_list = os.listdir(IR_path)
    VIS_image_list = os.listdir(VIS_path)

    IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))

    if args.num_test > 0:
        IR_image_list = IR_image_list[:args.num_test]
        VIS_image_list = VIS_image_list[:args.num_test]

    print(f"测试图像对数: {len(IR_image_list)}")

    to_tensor = transforms.ToTensor()

    # ========================================
    # 4. 消融实验 - 只保存融合图片
    # ========================================
    print("\n[3/3] 开始生成融合图片...")

    # 遍历每个配置
    for cfg in ablation_configs:
        print(f"\n{'='*60}")
        print(f"处理配置: {cfg['name']}")
        print(f"  SA: {'✓' if cfg['use_sa'] else '✗'}  |  CA: {'✓' if cfg['use_ca'] else '✗'}")
        print(f"  权重: {cfg['weight_path']}")
        print(f"{'='*60}")

        # 检查权重文件
        if not os.path.exists(cfg['weight_path']):
            print(f"⚠️  权重文件不存在，跳过此配置")
            continue

        # 加载 LAN
        SAR = Spatial_Aware_Refinement(
            spatial_weight=args.spatial_weight,
            use_sa=cfg['use_sa'],
            use_ca=cfg['use_ca']
        ).to(device).eval()

        SAR.load_state_dict(torch.load(cfg['weight_path'], map_location=device))
        print("✓ LAN 加载完成")

        # 创建保存目录
        save_dir = os.path.join(args.result_path, cfg['name'].replace(' ', '_').replace('/', '-'))
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 保存目录: {save_dir}")

        # 处理所有图像
        for idx, (ir_name, vis_name) in enumerate(tqdm(zip(IR_image_list, VIS_image_list),
                                                        total=len(IR_image_list),
                                                        desc=f"融合进度")):
            # 读取图像
            ir_path = os.path.join(IR_path, ir_name)
            vis_path = os.path.join(VIS_path, vis_name)

            img_ir = Image.open(ir_path).convert('L')
            img_vis = Image.open(vis_path).convert('RGB')

            ir = to_tensor(img_ir).unsqueeze(0).to(device)
            vis = to_tensor(img_vis).unsqueeze(0).to(device)

            # 融合
            fused = test_fusion(vis, ir, GLP, SAR, DNFN, device)

            # 保存融合结果
            fused_img = transforms.ToPILImage()(fused.squeeze(0).cpu())
            fused_img.save(os.path.join(save_dir, f'{idx+1}.png'))

        print(f"✅ {cfg['name']} 完成，共保存 {len(IR_image_list)} 张图像")

    # ========================================
    # 5. 完成总结
    # ========================================
    print("\n" + "="*80)
    print("🎉 所有消融实验融合完成！")
    print("="*80)
    print(f"\n📁 融合结果保存在: {args.result_path}")
    print(f"\n生成的文件夹:")
    for cfg in ablation_configs:
        folder_name = cfg['name'].replace(' ', '_').replace('/', '-')
        folder_path = os.path.join(args.result_path, folder_name)
        if os.path.exists(folder_path):
            num_images = len(os.listdir(folder_path))
            print(f"  ✓ {folder_name:20s} : {num_images} 张图像")

    print("\n" + "="*80)
    print("✅ 完成！您现在可以对比这些融合结果了。")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='简化版消融实验测试')

    # 数据路径
    parser.add_argument('--vis_dir', type=str,
                        default='./NestFuse/fusion_test_data/msrs/vi',
                        help='可见光图像目录')
    parser.add_argument('--ir_dir', type=str,
                        default='./NestFuse/fusion_test_data/msrs/ir',
                        help='红外图像目录')
    parser.add_argument('--result_path', type=str,
                        default='./test_results/sa_ca_msrs',
                        help='结果保存目录')

    # 模型路径
    parser.add_argument('--lfn_path', type=str,
                        default='./checkpoint/enhanced_train/LFN_epoch_20.pth',
                        help='LFN 模型权重路径')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./checkpoint/ablation_checkpoints',
                        help='消融模型权重目录')
    parser.add_argument('--dnfn_path', type=str,
                        default='./NestFuse/runs/train_12-04_16-01/checkpoints/epoch018-loss0.338.pth',
                        help='DNFN 模型权重路径')

    # 实验参数
    parser.add_argument('--spatial_weight', type=float, default=0.5,
                        help='空间注意力权重')
    parser.add_argument('--num_test', type=int, default=-1,
                        help='测试图像数量 (-1表示全部)')
    parser.add_argument('--save_images', action='store_true',
                        help='是否保存融合图像')

    args = parser.parse_args()
    main(args)