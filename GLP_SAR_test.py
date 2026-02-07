import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


# 导入你的模型类 (请确保文件名匹配)
from GLPN_model import Global_Luminance_Perception
from SARN_model import Spatial_Aware_Refinement

def test_inference(input_dir, save_dir, lfn_weights, lan_weights):
    # 1. 环境准备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 加载模型
    net_GLP = Global_Luminance_Perception().to(device)
    net_SAR = Spatial_Aware_Refinement().to(device)
    
    # 加载权重
    print(f"正在加载 LFN 权重: {lfn_weights}")
    net_GLP.load_state_dict(torch.load(lfn_weights, map_location=device))
    print(f"正在加载 LAN 权重: {lan_weights}")
    net_SAR.load_state_dict(torch.load(lan_weights, map_location=device))
    
    net_GLP.eval()
    net_SAR.eval()

    # 3. 图像预处理 (不带 Resize 以保持原图分辨率，如果显存不够可开启 Resize)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 4. 获取图片列表
    image_list = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_list:
        print(f"错误: 在 {input_dir} 中未找到图片。")
        return

    print(f"找到 {len(image_list)} 张待处理图片，开始推理...")

    with torch.no_grad():
        for img_name in image_list:
            # 读取图片
            img_path = os.path.join(input_dir, img_name)
            low_img_pil = Image.open(img_path).convert('RGB')
            w_orig, h_orig = low_img_pil.size
            
            input_tensor = transform(low_img_pil).unsqueeze(0).to(device) # (1, 3, H, W)

            # --- 执行推理 ---
            # 1. LFN 预测增益 (batch, 2)
            gain = net_GLP(input_tensor)
            
            # 2. LAN 增强
            # 注意：在 LAN 内部，gain[:, 0] 会被用来做计算
            # 返回: (中间结果, 最终结果, 调节参数x_r)
            _, enhanced_tensor, _ = net_SAR(input_tensor, gain)
            
            # --- 数据后处理 ---
            # 获取数值用于打印 (处理 batch=1 的情况)
            g_val = gain[0].cpu().numpy() # 获取 [g1, g2]
            
            # Tensor 转为 numpy 数组 (0-255, uint8)
            res_np = enhanced_tensor.squeeze(0).cpu().clamp(0, 1).numpy()
            res_np = (res_np.transpose(1, 2, 0) * 255).astype(np.uint8)
            res_pil = Image.fromarray(res_np)
            
            # --- 结果展示：将原图和结果图拼在一起 ---
            combined_res = Image.new('RGB', (w_orig * 2, h_orig))
            combined_res.paste(low_img_pil, (0, 0))      # 左侧原图
            combined_res.paste(res_pil, (w_orig, 0))     # 右侧增强图
            
            # 保存
            save_path = os.path.join(save_dir, f"res_{img_name}")
            combined_res.save(save_path)
            
            print(f"成功处理: {img_name} | Gain: {g_val}")

    print(f"\n所有图片处理完成！结果保存在: {save_dir}")

if __name__ == '__main__':
    # ================= 配置区 =================
    # 待测试的低光图片目录
    INPUT_DIR = './test_images/'       
    # 结果保存目录
    SAVE_DIR = './test_results1562/'
    # 训练好的模型路径 (根据你的实际文件名修改)
    # LFN_PATH = './checkpoint/train/LFN_stage2_epoch_11.pth'
    # LAN_PATH = './checkpoint/train/LAN_stage2_epoch_11.pth'
    # ==========================================
    LFN_PATH = './checkpoint/enhanced_train/LFN_epoch_20.pth'
    LAN_PATH = './checkpoint/enhanced_train/LAN_epoch_20.pth'
    test_inference(INPUT_DIR, SAVE_DIR, LFN_PATH, LAN_PATH)