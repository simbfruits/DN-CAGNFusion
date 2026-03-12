import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
import os.path as osp
from torchvision import transforms
from PIL import Image

to_tensor = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(1143)


class fusion_dataset_loader_test(data.Dataset):
    def __init__(self,data_dir,transform=to_tensor):
        super().__init__()
        dirname=os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'vi':
                self.vis_path=osp.join(temp_path)
        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform
    def __getitem__(self,index):
        name = self.name_list[index]  # 获得当前图片的名称
        inf_image = Image.open(os.path.join(self.inf_path, name))
        vis_image = Image.open(os.path.join(self.vis_path, name))
        ir_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        return vis_image, ir_image, name
    def __len__(self):
        return len(self.name_list)

def rgb2ycbcr(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y=torch.clamp(Y, min=0., max=1.0)
    Cr=torch.clamp(Cr, min=0., max=1.0)
    Cb=torch.clamp(Cb, min=0., max=1.0)
    Y = torch.unsqueeze(Y, 1)#升维
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    # temp = torch.cat((Y, Cr, Cb), dim=1) CPU版本
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (   temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,)
        .transpose(1, 3)
        .transpose(2, 3))
    return out

def ycbcr2rgb(input_im):
    B, C, W, H = input_im.shape
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]])
    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3).cuda()
    out = torch.clamp(out, min=0., max=1.0)
    return out
def clahe(image, batch_size):
    image = image.cpu().detach().numpy()  # 移至CPU中（转为Tensor），再转为NumPy
    results = []
    for i in range(batch_size):
        img = np.squeeze(image[i:i+1, :, :, :])
        out = np.array(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), dtype='uint8')
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
        result = clahe.apply(out)[np.newaxis][np.newaxis]
        results.append(result)
    results = np.concatenate(results, axis=0)
    image_hist = (results / 255.0).astype(np.float32)
    image_hist = torch.from_numpy(image_hist).cuda()
    return image_hist