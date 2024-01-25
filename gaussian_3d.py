import os
import itertools
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import pandas as pd
from configs.config_setting import setting_config

def get_label(label_path,name):
    data_label = pd.read_excel(label_path)
    result = data_label[data_label['姓名'] == name].index.tolist()
    array_2d = np.zeros((setting_config.num_classes, 3))
    if len(result) == 1:
        row_index = result[0]
        for i,k in enumerate(setting_config.label_num):
            x_value =data_label.loc[row_index + k, 'x']
            y_value = data_label.loc[row_index + k, 'y']
            z_value = data_label.loc[row_index + k, 'z'] + 100
            array_2d[i] = [x_value, y_value, z_value]
    else:
        print(f"找不到 {name}")
    array_2d=np.expand_dims(array_2d, axis=2)
    return array_2d

label_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/三维坐标表格.xlsx"
data_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/train/images/"
test_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/val/images/"
images_list = sorted(os.listdir(data_path))
testlist= sorted(os.listdir(test_path))
xyz = np.array(0)
test = np.array(0)
for i,img_path in enumerate(images_list):
    msk = get_label(label_path, img_path)
    if i == 0:
        xyz = msk.copy()
    else:
        xyz = np.dstack((xyz, msk))

n = xyz.shape[0]
Xn = xyz.transpose(0, 2, 1)

for i,img_path in enumerate(testlist):
    msk = get_label(label_path, img_path)
    if i == 0:
        test = msk.copy()
    else:
        test = np.dstack((test, msk))

nt = test.shape[0]
testXn = test.transpose(0, 2, 1)

# 定义一个函数来计算三维高斯分布
def gaussian_3d(x, y, z, mean_x, mean_y, mean_z, sigma):
    coeff = 1 / (sigma * np.sqrt(2 * np.pi))
    gauss_x = coeff * np.exp(-0.5 * ((x - mean_x) / sigma) ** 2)
    gauss_y = coeff * np.exp(-0.5 * ((y - mean_y) / sigma) ** 2)
    gauss_z = coeff * np.exp(-0.5 * ((z - mean_z) / sigma) ** 2)
    return gauss_x * gauss_y * gauss_z

points = np.squeeze(Xn[0,:,:])

# 定义标准差
sigma = 1.0

# 对每个坐标点赋予一个高斯分布
# gaussians = []
# for point in points:
#     mean_x, mean_y, mean_z = point
#     gaussian = lambda x, y, z: gaussian_3d(x, y, z, mean_x, mean_y, mean_z, sigma)
#     gaussians.append(gaussian)


import numpy as np
from scipy.stats import multivariate_normal

# 计算 KL 散度
def calculate_KL_divergence(mean1, cov1, mean2, cov2):
    cov2_inv = np.linalg.inv(cov2)
    kl_divergence = 0.5 * (np.trace(np.dot(cov2_inv, cov1)) +
                           np.dot(np.dot((mean2 - mean1), cov2_inv), (mean2 - mean1)) -
                           3 + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
    return kl_divergence

# Example usage
mu1 = np.array([84,42, 108])
cov1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

mu2 = np.array([74,42, 108])
cov2 = np.array([[1, 0, 0], [0, 27, 0], [0, 0, 30]])

kl_div = calculate_KL_divergence(mu1, cov1, mu2, cov2)
print("KL Divergence:", kl_div)
