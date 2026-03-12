from PIL import Image
import os

# 指定图片所在的文件夹路径
image_folder = "E:\pycharm\Project\Image-Fusion-main\/NestFuse\/fusion_result\LLVIP\pair"  # 将这里替换为实际的文件夹路径

# 遍历文件夹中的文件
for file_name in os.listdir(image_folder):
    file_path = os.path.join(image_folder, file_name)
    try:
        if os.path.isfile(file_path) and (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.png') or file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.bmp')):
            img = Image.open(file_path)
            num_channels = len(img.getbands())
            print(f"图片 {file_name} 的通道数为: {num_channels}")
    except:
        print(f"处理 {file_name} 时出现错误")