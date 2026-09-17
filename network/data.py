'''
Created by: Zhiqing Guo
Institutions: Xinjiang University
Email: guozhiqing@xju.edu.cn
Copyright (c) 2023
'''
from torch.utils.data import Dataset
from scipy import ndimage
import numpy as np
import cv2
from PIL import Image

# ---train&val data---
class SingleInputDataset(Dataset):
    def __init__(self, txt_path, train_transform=None, valid_transform=None):
        lines = open(txt_path, 'r')
        imgs = []
        labels = []  # ✅ 新增：保存标签列表
        for line in lines:
            line = line.rstrip()
            words = line.split()
            img_path, label = words[0], int(words[1])
            # imgs.append((words[0], int(words[1])))
            imgs.append((img_path, label))
            labels.append(label)  # ✅ 新增
            # labels.append(label)  # ✅ 新增

        self.imgs = imgs        # generate the global list
        self.labels = labels  # ✅ 新增
        self.train_transform = train_transform
        self.valid_transform = valid_transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        
        # transform
        if self.train_transform is not None:
            img = self.train_transform(img)
        if self.valid_transform is not None:
            img = self.valid_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
    
    def get_labels(self):  # ✅ 新增接口
        return np.array(self.labels)


# ---test data---
class TestDataset(Dataset):
    def __init__(self, txt_path, test_transform=None):
        lines = open(txt_path, 'r')
        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs        # generate the global list
        self.test_transform = test_transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        
        # transform
        if self.test_transform is not None:
            img = self.test_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
# class TestDataset(Dataset):
#     def __init__(self, txt_path, test_transform=None):
#         lines = open(txt_path, 'r')
#         imgs = []
#         for line in lines:
#             line = line.rstrip()
#             words = line.split()
#             img_path, label = words[0], int(words[1])
#             imgs.append((img_path, label))

#         self.imgs = imgs        # generate the global list
#         self.test_transform = test_transform

#     def __getitem__(self, index):
#         img_path, label = self.imgs[index]
#         img = Image.open(img_path).convert('RGB')
        
#         # transform
#         if self.test_transform is not None:
#             img = self.test_transform(img)

#         # ★★★ 关键！返回路径 ★★★
#         return img, label, img_path

#     def __len__(self):
#         return len(self.imgs)






