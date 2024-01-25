import pandas as pd
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt

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
# output = torch.tensor([[[[ -3,  3,  1]],
#
#          [[  2, 0,   4]],
#          [[ 5,  6,  2]]
#          ],
#
#
#         [[[ 5,  6,  2]],
#
#          [[ 3, 4,  3]],
#         [[  2, 0,   4]]]],
# )
#
# target=torch.tensor([[[[ 5,  6,  2]],
#
#          [[ 3, 4,  3]],
#
#          [[ -1,   5,  1]]],
#
#     [[[ -3,  3,  1]],
#
#          [[  2, 0,   4]],
#
#          [[1,  3,   3]]]]
#
# )

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
#
# import torch.nn as nn
#
# w=nn.Parameter(torch.ones(4))
# print(w[0])

import torch
import torch.nn as nn

# class Conv4DLayerNorm(nn.Module):
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.normalized_shape = normalized_shape
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         # 初始化可学习参数
#         if self.data_format == "channels_last":
#             self.weight = nn.Parameter(torch.ones(normalized_shape))
#             self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         elif self.data_format == "channels_first":
#             self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1, 1))
#             self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1, 1))
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, x.size()[1:], self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(dim=(2, 3, 4), keepdim=True)
#             s = (x - u).pow(2).mean(dim=(2, 3, 4), keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight * x + self.bias
#             return x
#
# conv_layer = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3,stride=4,padding=1,groups=16)
# input_tensor = torch.randn(4, 16, 128, 128)
# output_tensor = conv_layer(input_tensor)
# maxpool_layer = nn.MaxPool2d(kernel_size=4)
# output_tensor = maxpool_layer(output_tensor)
# # #layer_norm=nn.LayerNorm((64 ,128, 128, 4))
# # #out = layer_norm(output_tensor)
# # # ln = Conv4DLayerNorm(64,eps=1e-6, data_format="channels_first")
# # # output_tensor = ln(output_tensor)
# groupnorm3d = nn.GroupNorm(4,64)
# # # out=groupnorm3d(output_tensor)
# # # avg_pool = nn.AvgPool3d(kernel_size=2, stride=2)
# # # output_tensor = avg_pool(output_tensor)
# # out=F.gelu(output_tensor)
# # ap = nn.AdaptiveAvgPool3d((1, 1,3))
# # out = ap(output_tensor)
# print(output_tensor.shape)

# conv_layer = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=2,dilation=2)
# input_tensor = torch.randn(4, 16, 128, 128)
# output_tensor = conv_layer(input_tensor)
#
# print(output_tensor.shape)


# x=torch.tensor([[2,4],[1,3]])
# y=torch.tensor([[2,4],[43,3]])
# x1=torch.tensor([[2,4],[1,3]])
# z=torch.add(x,y,x1)
# print(z)

# import torch
#
# # 创建一个形状为(4, 64, 128, 128)的示例张量
# tensor = torch.randn(4, 64, 128, 128)
#
# # 将张量重新调整形状为(4, 16, 128, 128, 4)
# reshaped_tensor = tensor.view(tensor.shape[0],16,tensor.shape[2],tensor.shape[3],-1)
#
#
# print(reshaped_tensor.shape)  # 输出为 torch.Size([4, 16, 128, 128, 4])
#
# rs = reshaped_tensor.view(reshaped_tensor.shape[0],reshaped_tensor.shape[1]*reshaped_tensor.shape[4],reshaped_tensor.shape[2],reshaped_tensor.shape[3])
# print(rs.shape)

# weight = nn.Parameter(torch.ones(5))
# print(weight)

# input_tensor = torch.randn(4, 3, 1, 3)
# out = input_tensor.view(4,3,1,1,-1)
# print(out.shape)

#
# input_tensor = torch.randn(4, 16, 128, 128)
# out = torch.chunk(input_tensor,4,dim=1)
# print(np.array(out).shape)
# out1 = [torch.chunk(chunk,4,dim=2) for chunk in out]
#
# out11 = []
# for chunk in out1:
#     outt = []
#     for c in chunk:
#         outt.append(torch.chunk(c, 4, dim=3))
#     out11.append(outt)
#
#
# print(np.array(out1).shape)
# print(np.array(out11).shape)
#
# out111=[]
# for cat in out11:
#     outt = []
#     for c in cat:
#         outt.append(torch.cat(c,dim=3))
#     out111.append(outt)
# print(np.array(out1).shape)
# out2 = [torch.cat(cat,dim=2) for cat in out111]
# out3 = torch.cat(out2,dim=1)
#
# if torch.equal(out3,input_tensor):
#     print(out3.shape)
#
# first_name= "y"
# last_name ="xZ"
#
# full_name = f"{first_name} {last_name}"
# print(full_name)
#
# print(f"Hello ,{full_name.title()}!")

# print(bin(333))

# a= int(input())
# b= int(input())
#
# for i in range(a,b+1):
#     if '3' in str(i):
#         print(i)
# common=3
# listc = list(range(0, common, 1))
# randc = random.choice(listc)
# print(randc)

# import matplotlib.pyplot as plt
#
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
#
# plt.plot(x, y)
# plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
# plt.show()

# import torch
# import math
# # a = torch.tensor([[[[ -3,  3,  1]],[[  2, 0, 4]],[[ 5,  6,  2]]],[[[ 5,  6, 2]],[[ 3, 4,3]],[[2, 0,4]]]])
# # b = torch.tensor([[[[1,2,3]]]])
# #
# # c=a*b
# # print(c)
#
# print(torch.tensor(9.).norm(2))

# acc =[4,14,5]
# acc1 = [4,67,2]
#
# plt.plot(range(len(acc)), acc, label='no attention Acc')
# plt.plot(range(len(acc1)), acc1, label='attention Acc')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# # 添加图例
# plt.legend()
# # 显示图形
# plt.show()
#
# import torch
#
# tensor= torch.tensor([[[[-0.9505, -1.0060,  0.2246]],
#          [[ 1.1415,  1.7670, -0.1949]],
#          [[-0.7992,  0.0960, -0.0668]],
#          [[ 0.8869,  1.8635,  0.6075]]],
#         [[[ 0.1123,  1.8968,  0.3042]],
#          [[ 0.2816,  1.3720,  0.7798]],
#          [[ 1.6692, -0.9294,  0.4123]],
#          [[-0.9770, -0.0060, -0.3128]]],
#         [[[ 1.2612, -0.5297, -1.1499]],
#          [[-0.4561, -0.9753,  0.3849]],
#          [[ 1.3418, -1.4879, -1.5372]],
#          [[ 1.6373, -0.8774, -0.7387]]],
#         [[[ 0.6596,  0.5533, -1.2232]],
#          [[-1.4159, -0.6947,  0.9320]],
#          [[ 1.0285,  0.0119, -0.1490]],
#          [[-0.3919,  0.8766, -0.9656]]]])
#
# tensor1= torch.tensor([[[[ 0.6596,  0.5533, -1.2232]],
#          [[-1.4159, -0.6947,  0.9320]],
#          [[ 1.0285,  0.0119, -0.1490]],
#          [[-0.3919,  0.8766, -0.9656]]],
#         [[[-0.9505, -1.0060,  0.2246]],
#          [[ 1.1415,  1.7670, -0.1949]],
#          [[-0.7992,  0.0960, -0.0668]],
#          [[ 0.8869,  1.8635,  0.6075]]],
#         [[[ 0.1123,  1.8968,  0.3042]],
#          [[ 0.2816,  1.3720,  0.7798]],
#          [[ 1.6692, -0.9294,  0.4123]],
#          [[-0.9770, -0.0060, -0.3128]]],
#         [[[ 1.2612, -0.5297, -1.1499]],
#          [[-0.4561, -0.9753,  0.3849]],
#          [[ 1.3418, -1.4879, -1.5372]],
#          [[ 1.6373, -0.8774, -0.7387]]],
#         ])
#
# bi = torch.tensor([[0.4883, 0.4883, 0.5000],
#         [0.4924, 0.4924, 0.9964],
#         [0.4883, 0.4883, 1.0000],
#         [0.4883, 0.4883, 0.5000]])
#
# # out = (tensor-tensor1)*torch.tensor([[0.4883, 0.4883, 0.5000],
# #         [0.4924, 0.4924, 0.9964],
# #         [0.4883, 0.4883, 1.0000],
# #         [0.4883, 0.4883, 0.5000]])
# out=[]
# for i ,ten in enumerate(tensor):
#      ot = (ten-tensor1[i])*bi[i]
#      out.append(ot[:].tolist())
# print(torch.tensor(out).shape,out)

# # 沿着指定的轴求均值
# point_mean = torch.mean(tensor.float(), dim=2)
#
# print(point_mean)
# def find_positions(matrix, value):
#     positions = []
#     for i in range(len(matrix)):
#         for j in range(len(matrix[i])):
#             if matrix[i][j] < value:
#                 positions.append((i, j))
#     return positions
#
# result = find_positions(point_mean, 7)
# print(result)
# for t in result:
#      tensor[t[0]][t[1]] = torch.tensor([-1,-1,-1]).float()
# print(tensor)

# rate = np.full_like(np.array([1., 1., 1.]), 4.)
# f = np.array([1.1, 1.2, 1.3])
# print(rate*f)
# np1 = np.array([3.,1.,4.])
# ff = np.array([3.,1.,4.])
# print(np1*ff)

import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from configs.config_setting import setting_config


# def get_label(label_path,name):
#     data_label = pd.read_excel(label_path)
#     result = data_label[data_label['姓名'] == name].index.tolist()
#     array_2d = np.zeros((setting_config.num_classes, 3))
#     if len(result) == 1:
#         row_index = result[0]
#         for i,k in enumerate(setting_config.label_num):
#             x_value =data_label.loc[row_index + k, 'x']
#             y_value = data_label.loc[row_index + k, 'y']
#             z_value = data_label.loc[row_index + k, 'z'] + 100
#             array_2d[i] = [x_value, y_value, z_value]
#     else:
#         print(f"找不到 {name}")
#     return array_2d
#
# import numpy as np
# from sklearn.neighbors import KernelDensity
#
# def calculate_cross_entropy(x, kde_p, kde_q):
#     # 计算概率密度
#     p_density = np.exp(kde_p.score_samples(x.reshape(-1, 1)))
#     q_density = np.exp(kde_q.score_samples(x.reshape(-1, 1)))
#
#     # 将概率密度离散化
#     p_distribution = p_density / np.sum(p_density)
#     q_distribution = q_density / np.sum(q_density)
#
#     # 计算交叉熵
#     cross_entropy = -np.sum(p_distribution * np.log(q_distribution))
#
#     return cross_entropy
#
#
#
# # 使用蒙特卡罗积分来估算KL散度
# def monte_carlo_kl_divergence(kde_p, kde_q, samples):
#     # 使用kde_p的分布生成采样点
#     drawn_samples = kde_p.resample(size=samples).T
#     # 评估两个核密度估计在采样点处的概率密度值
#     p_vals = kde_p(drawn_samples.T)
#     q_vals = kde_q(drawn_samples.T)
#
#     # 计算KL散度的定义公式的一部分：p(x) * log(p(x)/q(x))
#     nonzero = q_vals > 0
#     divergences = p_vals[nonzero] * np.log(p_vals[nonzero] / q_vals[nonzero])
#     return np.mean(divergences)
#
#
# label_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/三维坐标表格.xlsx"
# data_patht = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/train/images/"
# data_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/val/images/"
# images_listt = sorted(os.listdir(data_patht))
# datast = np.zeros((len(images_listt),setting_config.num_classes,3))
# images_list = sorted(os.listdir(data_path))
# datas = np.zeros((len(images_list),setting_config.num_classes,3))
# for i, img_path in enumerate(images_list):
#     msk = get_label(label_path, img_path)
#     datas[i] = msk
# for i, img_path in enumerate(images_listt):
#     mskt = get_label(label_path, img_path)
#     datast[i] = mskt
from scipy.stats import gaussian_kde

# for k in range(setting_config.num_classes):#setting_config.num_classes
#     data = datas[:,k,:]
#     datat = datast[:,k,:]
#     # 定义核密度估计模型
#     kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
#     kdet = KernelDensity(bandwidth=1.0, kernel='gaussian')
#     # 用样本数据拟合模型
#     kde.fit(data)
#     kdet.fit(datat)
#
#     # 实例化真实数据和预测数据的核密度估计
#     kde_real = gaussian_kde(data.T)
#     kde_pred = gaussian_kde(datat.T)
#
#     # 生成用于绘制密度的网格点
#     x, y, z = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
#     grid_points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
#
#     # 计算每个网格点的概率密度
#     log_density = kde.score_samples(grid_points)
#
#     # 将一维的概率密度转换为三维网格
#     density = np.exp(log_density).reshape(100, 100, 100)
#
#     # 画出三维图
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[:, 0], data[:, 1], data[:, 2], color='blue', alpha=0.5)
#     ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], c=density.flatten(), cmap='viridis', marker='.')
#     plt.show()
#
#     # 计算KL散度
#     kl_div = monte_carlo_kl_divergence(kde_real, kde_pred, samples=1000)
#     print(f"Estimated KL divergence: {kl_div}")
from scipy.stats import entropy
# data = datast[:,0,:]
# datat = datas[:,0,:]
# np.random.seed(0)
# # data = np.random.multivariate_normal(mean=[0, 0, 0], cov=np.eye(3), size=1000)
# # datat = np.random.multivariate_normal(mean=[0.5, 0.5, 0.5], cov=np.eye(3), size=1000)
# kde_real = gaussian_kde(data.T)
# kde_pred = gaussian_kde(datat.T)
# # 计算KL散度
# kl_div = monte_carlo_kl_divergence(kde_real, kde_pred, samples=len(images_listt))
# print(f"Estimated KL divergence: {kl_div}")

#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from scipy.stats import multivariate_normal
# import os
# import itertools
# import numpy as np
# import pyswarms as ps
# from pyswarms.utils.functions import single_obj as fx
# import pandas as pd
# from configs.config_setting import setting_config
# from scipy.optimize import curve_fit
#
# def get_label(label_path,name):
#     data_label = pd.read_excel(label_path)
#     result = data_label[data_label['姓名'] == name].index.tolist()
#     array_2d = np.zeros((setting_config.num_classes, 3))
#     if len(result) == 1:
#         row_index = result[0]
#         for i,k in enumerate(setting_config.label_num):
#             x_value =data_label.loc[row_index + k, 'x']
#             y_value = data_label.loc[row_index + k, 'y']
#             z_value = data_label.loc[row_index + k, 'z'] + 100
#             array_2d[i] = [x_value, y_value, z_value]
#     else:
#         print(f"找不到 {name}")
#     array_2d=np.expand_dims(array_2d, axis=2)
#     return array_2d
#
# label_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/三维坐标表格.xlsx"
# data_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/train/images/"
# images_list = sorted(os.listdir(data_path))
# xyz = np.array(0)
# for i,img_path in enumerate(images_list):
#     msk = get_label(label_path, img_path)
#     if i == 0:
#         xyz = msk.copy()
#     else:
#         xyz = np.dstack((xyz, msk))
#
#
# '''
# x = np.linspace(np.min(xyz) - 5, np.max(xyz) + 5, 100)
# y = np.linspace(np.min(xyz) - 5, np.max(xyz) + 5, 100)
# x, y = np.meshgrid(x, y)
# pos = np.empty(x.shape + (3,))
# pos[:, :, 0] = x
# pos[:, :, 1] = y
# pos[:, :, 2] = 0  # Z维的值，由于高斯函数是关于xy平面对称的，我们可以取一个z的切面
#
# # 绘制高斯分布图
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
#
# for point in range(xyz.shape[0]):
#     points = np.squeeze(xyz.transpose(0, 2, 1)[point, :, :])
#     # 分别计算 x, y, z 的均值
#     mean_x, mean_y, mean_z = points.mean(axis=0)
#
#     # 分别计算 x, y, z 的标准差
#     std_x, std_y, std_z = points.std(axis=0)
#
#     # 输出均值和标准差
#     print("均值：", mean_x, mean_y, mean_z)
#     print("标准差：", std_x, std_y, std_z)
#
#     # 假设的均值和标准差
#     mean = [mean_x, mean_y, mean_z]
#     # 根据标准差创建对角线方差元素
#     var_x = std_x**2
#     var_y = std_y**2
#     var_z = std_z**2
#     # 创建协方差矩阵
#     covariance = np.array([
#         [var_x, 0, 0],
#         [0, var_y, 0],
#         [0, 0, var_z]
#     ])
#
#     # 多变量正态分布
#     rv = multivariate_normal(mean, covariance)
#
#     # 对于每一个x和y，计算概率密度函数的值
#     z = rv.pdf(pos)
#     ax.plot_surface(x, y, z, cmap=cm.viridis)
#
# # 设置图表标题和坐标轴标签
# ax.set_title('3D Gaussian Distribution')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Probability Density')
# # 显示图表
# plt.show()
# '''
#
# data = np.squeeze(xyz.transpose(0, 2, 1)[0, :, :])
# # 将Numpy数组转换为字符串表示，并在每个数字之后添加逗号
# np_str_with_commas = np.array2string(data, separator=', ')[1:-1]
# # 使用replace去掉最后的逗号并打印结果
# np_str_with_commas = np_str_with_commas.replace('],', '];')
# print(np_str_with_commas)
#

import cv2
import numpy as np
import random

# 读取原始图片
image = cv2.imread('/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/train/images/曹骄阳（张）正颌2/CT.1.3.46.670589.33.1.63778281912594160400001.5551689432045871359.png')

def noisejy(image):
    # 获取图片的高度和宽度
    height, width, channels = image.shape

    # 添加噪声的比例（可以根据需要调整）
    noise_ratio = 0.02

    # 添加椒盐噪声
    num_noise_pixels = int(noise_ratio * height * width)
    for _ in range(num_noise_pixels):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        value = random.randint(0, 255)
        if random.random() < 0.5:
            image[y, x] = [value, value, value]  # 盐噪声
        else:
            image[y, x] = [0, 0, 0]  # 椒噪声
    return image

def add_gaussian_noise(image, mean, stddev):
    # 生成随机噪声
    noise = np.random.randn(*image.shape) * stddev + mean
    # 将噪声添加到图像像素上
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def hsvchange(image):
    brightness_factor = 0.5
    contrast_factor = 0.5
    darkened_image = np.int16(image)
    darkened_image = darkened_image * contrast_factor + brightness_factor
    darkened_image = np.clip(darkened_image, 0, 255).astype(np.uint8)
    return darkened_image

# image = add_gaussian_noise(image, mean=0, stddev=20)
# cv2.imshow("srcimg",image)
# image = hsvchange(image)

# import numpy as np
#
# # 输入的一维数组
# arr = np.array([1, 2, 4])
#
# # 将一维数组转换为对角矩阵
# diag_matrix = np.diag(arr)
#
# # 输出对角矩阵
# print(diag_matrix)
#
# cv2.imshow("noise img ",image)
# cv2.waitKey()

# import torch
# from torch.distributions.normal import Normal
#
# # 假设模型输出是正态分布的均值和标准差
# model_mu = torch.tensor([1.0, 2.0, 3.0])
# model_sigma = torch.tensor([0.1, 0.2, 0.3])
#
# # 真实标签的正态分布参数
# true_mu = torch.tensor([1.2, 2.1, 2.8])
# true_sigma = torch.tensor([0.15, 0.18, 0.25])
#
# # 构建正态分布对象
# model_dist = Normal(loc=model_mu, scale=model_sigma)
# true_dist = Normal(loc=true_mu, scale=true_sigma)
#
# # 计算 KL 散度
# kl_divergence = torch.distributions.kl.kl_divergence(model_dist, true_dist).sum()
# print(kl_divergence)
