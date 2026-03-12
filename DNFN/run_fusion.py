# -*- coding: utf-8 -*-
"""
Writer: ZZQ
Date: 2024 02 22
"""
import os
import re
from torchvision.utils import save_image
from utils.util_device import device_on
from util_fusion import image_fusion
# False
defaults = {
    "gray": True,
    "deepsupervision": True,
    "model_name": 'NestFuse_eval',
    # "model_weights": 'runs/train_11-19_16-41/checkpoints/epoch009-loss0.001.pth',#initial
    # "model_weights": 'runs/train_11-29_11-51_2model/checkpoints/epoch019-loss0.322.00012.pth',#perfect
    "model_weights": 'runs/train_12-04_16-01/checkpoints/epoch018-loss0.338.pth',
    # "model_weights": 'runs/train_12-06_16-14dense_bn/checkpoints/epoch018-loss0.333.pth',

    "device": device_on(),
}
''' 
/****************************************************/
    main
/****************************************************/
'''
if __name__ == '__main__':
    fusion_instance = image_fusion(defaults)
    # ---------------------------------------------------#
    #   单对图像融合
    # ---------------------------------------------------#
    # if True:
    #     image1_path = "fusion_test_data/Tno/IR_images/IR3.png"
    #     image2_path = "fusion_test_data/Tno/VIS_images/VIS3.png"
    #     result_path = 'fusion_result/pair'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     Fusion_image = fusion_instance.run(image1_path, image2_path)
    #     save_image(Fusion_image, f'{result_path}/fused_image2.png')

    # ---------------------------------------------------#
    #   多对图像融合
    # ---------------------------------------------------#
    # IR_path = "fusion_test_data/fire_data/Region_thermal"
    # VIS_path = "fusion_test_data/fire_data/Region_visible"
    # result_path = 'fusion_result/fusion_result_fire'
    IR_path = "fusion_test_data/M3FD/ir"
    VIS_path = "fusion_test_data/M3FD/vi"
    result_path = 'fusion_result/M3FD/1'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print('载入数据...')
    IR_image_list = os.listdir(IR_path)
    VIS_image_list = os.listdir(VIS_path)
    IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    print('开始融合...')
    num = 0
    for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
        num += 1
        IR_image_path = os.path.join(IR_path, IR_image_name)
        VIS_image_path = os.path.join(VIS_path, VIS_image_name)
        Fusion_image = fusion_instance.run(IR_image_path, VIS_image_path)
        save_image(Fusion_image, f'{result_path}/{num}.png')
        # print('输出路径：' + result_path + 'fusion{}.png'.format(num))
