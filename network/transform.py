# '''
# Created by: Zhiqing Guo
# Institutions: Xinjiang University
# Email: guozhiqing@xju.edu.cn
# Copyright (c) 2023
# '''
# from PIL import Image,ImageFilter
# from torchvision import transforms
# import  albumentations as A
# import numpy as np
# import cv2
# albumentations_transform = A.Compose([
#     A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
#     A.GaussNoise(p=0.1),
#     # A.GaussianBlur(blur_limit=3, p=0.05),
#     A.GaussianBlur(blur_limit=(3, 7), p=0.05),
#     A.HorizontalFlip(),
#     # 新增
#     A.CLAHE(p=0.5),
#     A.RandomGamma(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
#     # A.RandomGridShuffle(grid=(3, 3), p=0.3),
#     # A.cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),
#     # A.cutout(n_holes=8, max_h_size=16, max_w_size=16, fill_value=0, p=0.2),

#     # A.ImageCompression(quality_lower=60, quality_upper=90, p=0.5)，
#     # A.Resize(height=256, width=256, always_apply=True),  # 等比例缩放
#     # A.PadIfNeeded(min_height=256, min_width=256, border_mode=cv2.BORDER_CONSTANT),
#     A.OneOf([
#         A.RandomBrightnessContrast(),
#         A.FancyPCA(),
#         A.HueSaturationValue()
#     ], p=0.7),
#     A.ToGray(p=0.2),
#     # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
# ])

# class AlbumentationsTransform:
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, img):
#         img = np.array(img)
#         img = self.transform(image=img)['image']
#         return Image.fromarray(img)
# class RandomBlur(object):
#     def __init__(self, radius=1):
#         self.radius = radius
 
#     def __call__(self, img):
#         return img.filter(ImageFilter.GaussianBlur(self.radius))
# class AugMix(object):
#     def __init__(self, alpha=1.0, num_ops=3):
#         self.alpha = alpha
#         self.num_ops = num_ops
#         self.operations = [
#             transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
#             # transforms.RandomRotation(degrees=10),
#             # transforms.RandomHorizontalFlip(),
#             # transforms.RandomVerticalFlip(),
#             # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#             # transforms.RandomResizedCrop(size=24, scale=(0.8, 1.0)),
#             RandomBlur(radius=1)
#         ]
 
#     def __call__(self, img):
#         img = Image.fromarray(np.array(img))
#         mixed_img = img.copy()
        
#         for _ in range(self.num_ops):
#             op = np.random.choice(self.operations)
#             augmented_img = op(img)
#             if np.random.rand() < 0.5:
#                 mixed_img = Image.blend(mixed_img, augmented_img, alpha=self.alpha)
#             else:
#                 mixed_img = Image.blend(mixed_img, augmented_img, alpha=1 - self.alpha)
                
#         return mixed_img

# Data_Transforms = {
#     'train': transforms.Compose([
#         transforms.Resize((224, 224)),
#         #transforms.ToPILImage(),
#         # transforms.Resize((256,256)),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),# ColorJitter 后续添加
#         AlbumentationsTransform(albumentations_transform),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         # AugMix(alpha=0.3, num_ops=3),
#         # transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),    # Rotate in (-10,10)
#         transforms.RandomPerspective(),
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#         # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize((224, 224)),
#         # transforms.Resize((256,256)),
#         #transforms.ToPILImage(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ]),
#     'test': transforms.Compose([
#         transforms.Resize((224, 224)),
#         # transforms.Resize((256,256)),
#         #transforms.ToPILImage(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ]),
# }
# from albumentations.pytorch import ToTensorV2
# class AlbumentationsTransform:
#     def __init__(self, transform):
#         self.transform = transform

#     def __call__(self, img):
#         img = np.array(img)  # PIL → ndarray
#         transformed = self.transform(image=img)
#         return transformed['image']  # Tensor

# # ✅ 只加高斯噪声的 albumentations 测试增强
# albumentations_test_transform = A.Compose([
#     A.Resize(224, 224),
#     A.GaussNoise(var_limit=(50.0, 70.0), p=1.0),  # 100% 加噪声，方差范围 10~30
#     A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0),
#     A.GaussianBlur(p=0.5),
#     A.ShiftScaleRotate(scale_limit=0.2, shift_limit=0.0, rotate_limit=0, p=1.0),
#     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
#     A.CoarseDropout(max_holes=8, max_height=32, max_width=32,
#                 min_holes=1, min_height=8, min_width=8,
#                 fill_value=0, mask_fill_value=None, p=0.5),
#     # 

#     # A.Rotate(limit=60, p=1.0),
#     A.Normalize(mean=(0.485, 0.456, 0.406),
#                 std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])
# # Data_Transforms = {
# #     'train': transforms.Compose([
# #         transforms.Resize((224,224)),
# #         # transforms.ToPILImage(),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.RandomRotation(10),    # Rotate in (-10,10)
# #         transforms.RandomPerspective(),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                              std=[0.229, 0.224, 0.225])
# #     ]),
# #     'val': transforms.Compose([
# #         transforms.Resize((224,224)),
# #         #transforms.ToPILImage(),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                              std=[0.229, 0.224, 0.225])
# #     ]),
# #     # 
# #     'test': AlbumentationsTransform(albumentations_test_transform) 
# # }

# # Data_Transforms = {
# #     'train': transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         #transforms.ToPILImage(),
# #         transforms.RandomHorizontalFlip(),
# #         transforms.RandomRotation(10),    # Rotate in (-10,10)
# #         transforms.RandomPerspective(),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                              std=[0.229, 0.224, 0.225])
# #     ]),
# #     'val': transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         #transforms.ToPILImage(),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                              std=[0.229, 0.224, 0.225])
# #     ]),
# #     'test': transforms.Compose([
# #         transforms.Resize((224, 224)),
# #         #transforms.ToPILImage(),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                              std=[0.229, 0.224, 0.225])
# #     ]),
# # }

import random
import math
import numpy as np
from PIL import Image, ImageFilter
import albumentations as A
import torch
from torchvision import transforms
import cv2

# 定义 RandomErasing 类
class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            width, height = img.size  # 获取宽度和高度

            area = width * height

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x1 + w), int(y1 + h)

                if img.mode == 'RGB':
                    img.paste((int(self.mean[0]), int(self.mean[1]), int(self.mean[2])), (x1, y1, x2, y2))
                else:
                    img.paste(self.mean[1], (x1, y1, x2, y2))

                return img

        return img


# 定义一个增强管道类，结合 Albumentations 和 RandomErasing
class AlbumentationsTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return Image.fromarray(img)


# 定义数据增强
albumentations_transform = A.Compose([
    A.ImageCompression(quality_range=(60, 100), p=0.5),
    A.GaussNoise(p=0.1),
    A.GaussianBlur(blur_limit=(3, 7), p=0.05),
    A.HorizontalFlip(),
    A.CLAHE(p=0.5),
    A.RandomGamma(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.FancyPCA(),
        A.HueSaturationValue()
    ], p=0.7),
    A.ToGray(p=0.2),
])

# 将 RandomErasing 和 AlbumentationsTransform 结合到一起
class CombinedTransform:
    def __init__(self, alb_transform, random_erasing):
        self.alb_transform = alb_transform
        self.random_erasing = random_erasing

    def __call__(self, img):
        img = self.alb_transform(img)  # 首先应用 Albumentations 增强
        img = self.random_erasing(img)  # 然后应用 RandomErasing 增强
        return img


# 创建数据增强对象
random_erasing = RandomErasing(EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3)
combined_transform = CombinedTransform(AlbumentationsTransform(albumentations_transform), random_erasing)

# 定义训练数据增强管道
Data_Transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        combined_transform,  # 将自定义的增强方法加入到数据转换流程中
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # combined_transform  # 将自定义的增强方法加入到数据转换流程中
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        # A.GaussNoise(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
}
