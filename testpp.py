import pandas as pd
import numpy as np
import os
import cv2
import torch

# # 读取 Excel 文件
# data = pd.read_excel('/media/yxz/Elements/三维坐标表格.xlsx')
# dicom_dir = "/media/yxz/Elements/val/images"
# for j in sorted(os.listdir(dicom_dir)):
#     r_path = os.path.join(dicom_dir, j)
#     # 在 "姓名" 列中查找某个名字
#     name_to_find = j
#     result = data[data['姓名'] == name_to_find].index.tolist()
#     if len(result) == 1:
#         row_index = result[0]
#         array_2d = np.zeros((3, 3))
#         for i in range(3):
#             x_value = data.loc[row_index+i, 'x']
#             y_value = data.loc[row_index+i, 'y']
#             z_value = data.loc[row_index+i, 'z']
#             array_2d[i]=[x_value,y_value,z_value]
#
#             if pd.isnull(x_value) or pd.isnull(y_value) or pd.isnull(z_value):
#                 print(j)
#                 break
#         # print(array_2d)
#     else:
#         print(f"找不到 {name_to_find}")

# def normalize(img,mean,std):
#     img_normalized = (img - mean) / std
#     img_normalized = ((img_normalized - np.min(img_normalized))
#                       / (np.max(img_normalized) - np.min(img_normalized)))*255.
#     return img_normalized

#img = cv2.imread("/media/yxz/Elements/train/images/曹剑（沈）正颌/CT.1.3.46.670589.33.1.63816808395857726800001.5597244408443114737.png")
# 将图像的数据类型转换为浮点型
# 生成一个全零的 3x3x1 数组

# img = np.zeros((2, 2, 3))
# print(img)
#
# img = img.astype(np.float32)
#
# img=normalize(img,(0.5,0.4,0.3),(0.4,0.2,0.1))
# # img=torch.from_numpy(img).permute(2,0,1)
# print(img)

# def flip_point_horizontally(point, image_width):
#     flipped_x = image_width - point[..., 0]
#     flipped_point = torch.cat((flipped_x.unsqueeze(-1), point[..., 1:]), dim=-1)
#     return flipped_point
# point = torch.rand((1, 3, 3))
# print(point)
# w=2
# print(flip_point_horizontally(point,w))

import torch
import torchvision.transforms as transforms

# class RandomVerticalFlip(object):
#     def __init__(self, y_dim, p=1):
#         self.p = p
#         self.y_dim = y_dim
#
#     def __call__(self, point):
#         #if torch.rand(1) < selp:
#
#         point[..., 1] = self.y_dim - point[..., 1]
#
#         return point
#
# point = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])  # 示例点张量
# y_dim = 3  # Y 坐标的最大值，根据实际情况进行设置
#
# # 创建随机垂直翻转变换
# v_flip = RandomVerticalFlip(y_dim)
# print("原始点坐标:\n", point)
# # 对点张量进行变换
# flipped_point = v_flip(point)
#
# print("随机垂直翻转后的点坐标:\n", flipped_point)

import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
import math
import random

# class RandomRotation(object):
#     def __init__(self, angle_range, p=0.5):
#         self.angle_range=angle_range
#         self.p = p
#
#     def __call__(self, point):
#         if torch.rand(1) < self.p:
#             angled =  random.uniform(self.angle_range[0], self.angle_range[1])
#             print(angled)
#          # 定义旋转角度（以弧度为单位）
#             angle_degrees = angled
#             # angle_radians = math.radians(angle_degrees)
#             # angle = torch.tensor([angle_radians])
#             angle=torch.deg2rad(torch.tensor([30]))
#             # 提取前两个维度，即 x 和 y 坐标
#             xy = point[..., :2]
#             # 计算旋转后的新坐标
#             print(torch.cos(angle))
#             x_new = xy[..., 0] * torch.cos(angle) - xy[..., 1] * torch.sin(angle)
#             y_new = xy[..., 0] * torch.sin(angle) + xy[..., 1] * torch.cos(angle)
#
#             # 将旋转后的坐标保存回原始张量
#             point[..., 0] = x_new
#             point[..., 1] = y_new
#
#         return point
#
# point = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])  # 示例点张量
# angle_range = (-45, 45)  # 旋转角度范围
#
# # 创建随机旋转变换
# rotation = RandomRotation(angle_range, p=1)
# print("原始点坐标:\n", point)
# # 对点张量进行变换
# rotated_point = rotation(point)
# print("随机旋转后的点坐标:\n", rotated_point)


# import torch
#
# # 定义旋转角度（以弧度为单位）
#
# theta = torch.deg2rad(torch.tensor([30]))  # 将角度转换为弧度
# # 定义原始向量
# point = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
#
# # 提取前两个维度，即 x 和 y 坐标
# xy = point[..., :2]
#
# # 计算旋转后的新坐标
# x_new = xy[..., 0] * torch.cos(theta) - xy[..., 1] * torch.sin(theta)
# y_new = xy[..., 0] * torch.sin(theta) + xy[..., 1] * torch.cos(theta)
#
# # 将旋转后的坐标保存回原始张量
# point[..., 0] = x_new
# point[..., 1] = y_new
#
# # 输出旋转后的结果
# print(point)
#
#

# mask = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
# mask[...,0] = mask[...,0]*(50/30)
# # mask[...,1] = mask[...,1]*(40/image.shape[2])
# # print(mask)
output = torch.tensor([[[[ -3,  3,  1]],

         [[  2, 0,   4]],
         [[ 5,  6,  2]]
         ],


        [[[ 5,  6,  2]],

         [[ 3, 4,  3]],
        [[  2, 0,   4]]]],
)

target=torch.tensor([[[[ 5,  6,  2]],

         [[ 3, 4,  3]],

         [[ -1,   5,  1]]],

    [[[ -3,  3,  1]],

         [[  2, 0,   4]],

         [[1,  3,   3]]]]

)

#
# loss=torch.sqrt((output - target)**2)
# loss=torch.sum(target,dim=(2,3))
# cout=(loss[1] <= 6).sum().item()
# print(loss,cout)

# print(math.sqrt(3*(3**2)))

# size = output.size(0)
# print(size)
# pred_ = output.view(size, -1)
# print(pred_)
# target_ = target.view(size, -1)
# print(target_)

import torch.nn as nn

w=nn.Parameter(torch.ones(4))
print(w[0])

