import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from models import fuse_model, fusion_layer

import dataloader
# from NestFuse.dataloader import rgb2ycbcr , ycbcr2rgb, clahe


'''
/****************************************************/
    模型推理
/****************************************************/
'''


class image_fusion():
    # ---------------------------------------------------#
    #   初始化
    # ---------------------------------------------------#
    def __init__(self, defaults, **kwargs):
        """
        初始化方法
        :param defaults: 一个字典，包含模型的默认配置
        :param kwargs: 关键字参数，用于覆盖或添加默认配置
        """
        self.__dict__.update(defaults)  # 更新实例的属性为传入的默认配置
        for name, value in kwargs.items():
            setattr(self, name, value)  # 更新或添加属性
        # ---------------------------------------------------#
        #   载入预训练模型和权重
        # ---------------------------------------------------#
        self.load_model()

    def load_model(self):
        # ---------------------------------------------------#
        #   创建模型
        # ---------------------------------------------------#
        in_channel = 1 if self.gray else 3
        out_channel = 1 if self.gray else 3
        deepsupervision = self.deepsupervision
        self.model = fuse_model(self.model_name, input_nc=in_channel, output_nc=out_channel,
                                deepsupervision=deepsupervision)
        # ----------------------------------------------------#
        #   device
        # ----------------------------------------------------#
        device = self.device
        # ----------------------------------------------------#
        #   载入模型权重
        # ----------------------------------------------------#
        self.model = self.model.to(device)
        checkpoint = torch.load(self.model_weights, map_location=device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        print('{} model loaded.'.format(self.model_weights))

    def preprocess_image(self, image_path):
        # 读取图像并进行处理
        image = read_image(image_path, mode=ImageReadMode.GRAY if self.gray else ImageReadMode.RGB)

        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                               transforms.ToTensor(),
                                               ])

        image = image_transforms(image).unsqueeze(0)
        return image

    def run(self, image1_path, image2_path):
        self.model.eval()
        with torch.no_grad():
            # num_channels = image2_path.shape[0] if len(image2_path.shape) == 3 else image2_path.shape[1]
            # print(f"图像image2_path的通道数为: {num_channels}")
            print(image1_path)
            image1 = self.preprocess_image(image1_path).to(self.device)
            image2 = self.preprocess_image(image2_path).to(self.device)
            # image2 = self.preprocess_image_RGB(image2_path).to(self.device)
            # num_channels = image1.shape[0] if len(image1.shape) == 3 else image1.shape[1]
            # print(f"图像image1的通道数为: {num_channels}")
            # image_vis_ycbcr = rgb2ycbcr(image2)
            # image_vis_y = NestFuse.dataloader.clahe(image_vis_ycbcr[:, 0:1, :, :], image_vis_ycbcr.shape[0])
            # print("image_vis_y",image_vis_y.shape)
            image1_EN = self.model.encoder(image1)
            image2_EN = self.model.encoder(image2)
            # image2_EN = self.model.encoder(image_vis_y)
            # 进行融合
            # Fusion_image_feature = self.process_images(image1_EN, image2_EN, image2, image_vis_ycbcr,image_vis_y)#256415649865123
            Fusion_image_feature = fusion_layer(image1_EN, image2_EN)
            # print("Fusion_image_feature",Fusion_image_feature)
            if not self.deepsupervision:
                # 进行解码
                Fused_image = self.model.decoder(Fusion_image_feature)
                # 张量后处理
                Fused_image = Fused_image.detach().cpu()
                Fused_image = Fused_image[0]
            else:
                # 进行解码
                Fused_image = self.model.decoder(Fusion_image_feature)
                # 张量后处理
                # Fused_image = Fused_image[0].detach().cpu()
                # Fused_image = Fused_image[1].detach().cpu()
                Fused_image = Fused_image[2].detach().cpu()
                Fused_image = Fused_image[0]
        return Fused_image

    # 类方法是属于类而不是实例的方法，它可以通过类本身调用，也可以通过类的实例调用。
    # 类方法的特点是第一个参数通常被命名为cls，指向类本身，而不是指向实例。
    # 在类级别上操作或访问类属性，而不需要实例化对象
    @classmethod
    def get_defaults(cls, attr_name):
        """
        获取类的默认配置参数
        :param attr_name:接收一个参数attr_name，用于指定要获取对应配置属性的默认值
        :return:
        """
        if attr_name in cls._defaults:  # 首先检查 attr_name 是否在类属性 _defaults 中，如果在，则返回对应属性的默认值。
            return cls._defaults[attr_name]
        else:  # 如果 attr_name 不在 _defaults 中，则返回一个字符串，表示未识别的属性名称。
            return "Unrecognized attribute name '" + attr_name + "'"



def main():
    # 设置默认配置参数
    defaults = {
        "gray":True,
        "deepsupervision": True,
        "model_name": 'NestFuse_eval',
        # "model_weights": 'runs/train_11-19_16-41/checkpoints/epoch009-loss0.001.pth',
        "model_weights": 'runs/train_12-04_16-01new2_DenseBlock_cat/checkpoints/epoch017-loss0.358.pth',
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    # 创建image_fusion类的实例
    fusion_instance = image_fusion(defaults)

    # 调用load_model方法来加载模型
    fusion_instance.load_model()


if __name__ == "__main__":
    main()
    # 555555555555555555555555555555555555555555555555555555555

    # 22222222222222222222222222222222222222222222222222222222222
    # def preprocess_image_RGB(self, image_path):
    #     # 读取图像并进行处理
    #     image = read_image(image_path, mode=ImageReadMode.RGB)
    #
    #     image_transforms = transforms.Compose([transforms.ToPILImage(),
    #                                            transforms.ToTensor(),
    #                                            ])
    #
    #     image = image_transforms(image).unsqueeze(0)
    #     return image
    # 11111111111111111111111111111111111111111111111111111111111111111111111
    #     def process_images(self,image1_EN, image2_EN,image2,image_vis_ycbcr,image_vis_y):
    #         """
    #         根据输入的两张图像的通道数情况，执行不同的操作流程
    #
    #         :param image1_EN: 第一张图像对应的PyTorch张量
    #         :param image2_EN: 第二张图像对应的PyTorch张量
    #         :return: 根据图像通道数执行相应操作后得到的结果（可能是fusion_ycbcr或者Fusion_image_feature）
    #         """
    #         num_channels_image2 = image2.shape[0] if len(image2.shape) == 3 else image2.shape[1]
    #         if num_channels_image2 == 3:
    #             print("3通道")
    #             # image_vis_ycbcr = rgb2ycbcr(image2)
    #             # image_vis_y = NestFuse.dataloader.clahe(image_vis_ycbcr[:, 0:1, :, :], image_vis_ycbcr.shape[0])
    #             Y1, Y2, Y3, Y4 = fusion_layer(image1_EN, image_vis_y)
    #             print("Y4", Y4.shape)
    #             fusion_ycbcr1 = torch.cat((Y1, image_vis_ycbcr[:, 1:2, :, :], image_vis_ycbcr[:, 2:, :, :]), dim=1)
    #             print("fusion_ycbcr1",fusion_ycbcr1.shape)
    #             fusion_ycbcr2 = torch.cat((Y2, image_vis_ycbcr[:, 1:2, :, :], image_vis_ycbcr[:, 2:, :, :]), dim=1)
    #             fusion_ycbcr3 = torch.cat((Y3, image_vis_ycbcr[:, 1:2, :, :], image_vis_ycbcr[:, 2:, :, :]), dim=1)
    #             fusion_ycbcr4 = torch.cat((Y4, image_vis_ycbcr[:, 1:2, :, :], image_vis_ycbcr[:, 2:, :, :]), dim=1)
    #             Fusion_image_feature = [fusion_ycbcr1, fusion_ycbcr2, fusion_ycbcr3, fusion_ycbcr4]
    #
    #             return Fusion_image_feature
    #         elif num_channels_image2 == 1:
    #             print("1通道")
    #             Fusion_image_feature = fusion_layer(image1_EN, image2_EN)
    #             return Fusion_image_feature
    #         else:
    #             print("输入图像的通道数不符合预期，无法处理")
    #             return None
    # 11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111