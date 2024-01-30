import os
import itertools
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import pandas as pd
from configs.config_setting import setting_config

def get_label(label_path,name,kd=1.):
    data_label = pd.read_excel(label_path)
    result = data_label[data_label['姓名'] == name].index.tolist()
    array_2d = np.zeros((setting_config.num_classes, 3))
    if len(result) == 1:
        row_index = result[0]
        for i,k in enumerate(setting_config.label_num):
            x_value =data_label.loc[row_index + k, 'x']
            y_value = data_label.loc[row_index + k, 'y']
            z_value = kd*data_label.loc[row_index + k, 'z'] + 100
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

for i,img_path in enumerate(images_list):
    msk = get_label(label_path, img_path,-1.)
    xyz = np.dstack((xyz, msk))

n = xyz.shape[0]
Xn = xyz.transpose(0, 2, 1)**2


for i,img_path in enumerate(testlist):
    msk = get_label(label_path, img_path)
    if i == 0:
        test = msk.copy()
    else:
        test = np.dstack((test, msk))

nt = test.shape[0]
testXn = test.transpose(0, 2, 1)**2

def f_per_particle(m, Xn):
    """计算每个粒子的误差函数，使用`Xn`列表作为输入矩阵。"""
    # 划分粒子表示的变量（前n*9个元素表示a矩阵，剩余的3个表示b）
    ans = [m[i * 3:(i + 1) * 3].reshape(1, 3) for i in range(n)]
    b = m[n * 3:].reshape(1, 1)

    # 计算每个样本的误差
    loss = 0
    for sample in zip(*Xn):
        prediction = np.zeros((1, 1))
        for xi, an in zip(sample, ans):
            prediction += np.dot(an, xi.reshape(-1, 1))
        loss += np.sqrt(np.sum((prediction - b)**2))
    return loss

def f(x, Xn):
    """计算所有粒子的误差函数"""
    n_particles = x.shape[0]
    j = [f_per_particle(x[i], Xn) for i in range(n_particles)]
    return np.array(j)


# PSO参数设置
options = {'c1': 1.0, 'c2': 1.0, 'w': 0.8}
#options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# 粒子总数和维度（每个an有9个参数，b有3个参数，所以总共需要n*9+3个参数）
n_particles = 120
dimensions = n * 3 + 1
# 初始化全局最佳PSO，并传入bounds参数
#optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options)
# 运行优化器
#cost, pos = optimizer.optimize(f, iters=8000, Xn=Xn)
# 获取最优参数
# ans_opt = [pos[i * 3:(i + 1) * 3].reshape(1, 3) for i in range(n)]
# b_opt = pos[n * 3:].reshape(1, 1)

# 定义参数范围
c1_range = [0.5, 1.0, 1.5]
c2_range = [0.5, 1.0, 1.5]
w_range = [0.6, 0.8, 1.0]

# 组合参数范围
param_ranges = [c1_range, c2_range, w_range]
param_combinations = list(itertools.product(*param_ranges))

# 遍历每个参数组合并进行优化
best_cost = float('inf')
best_options = None

min_bound = np.full((dimensions,), 0.01)
max_bound = np.full((dimensions,), 1)
bounds = (min_bound, max_bound)

# for params in param_combinations:
#     options = {'c1': params[0], 'c2': params[1], 'w': params[2]}
#
#     # 初始化优化器并进行优化
#     optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options)
#     #cost, pos = optimizer.optimize(fx.sphere, iters=2000,Xn=Xn)
#     cost, pos = optimizer.optimize(f, iters=8000, Xn=Xn)
#
#     # 更新最佳参数设置
#     if cost < best_cost:
#         best_cost = cost
#         best_options = options
#         ans_opt = [pos[i * 3:(i + 1) * 3].reshape(1, 3) for i in range(n)]
#         b_opt = pos[n * 3:].reshape(1, 1)
#
#
# print("最佳参数设置:", best_options)
# print("最佳成本:", best_cost)
#
# # 打印结果
# print('Optimized ans:')
# for i, an_opt in enumerate(ans_opt, start=1):
#     print(f'a{i}_opt:\n{an_opt}')
# print('Optimized b:', b_opt)

def testx(n):
    a1_opt = np.array([-6.21007172e-05, 1.28758479e-06, -2.68045118e-04])
    a2_opt = np.array([-2.07947256e-04, 4.46834409e-05, 6.14944605e-06])
    a3_opt = np.array([-2.21353520e-04, 6.30610806e-05, 1.87035802e-04])
    a4_opt = np.array([2.54301043e-04, -9.25418365e-05, -7.99169087e-06])
    a5_opt = np.array([2.42138186e-04, 1.33552915e-04, 4.71636951e-05])
    a6_opt = np.array([4.30034321e-06, -1.38483871e-04, 7.74779972e-05])
    #b = 3.88585612e-03
    b=2

    # a1_opt = np.array([ 0.00100212 , 0.00093754, -0.0032127 ])
    # a2_opt = np.array([-0.00030611 ,-0.00039637 , 0.00240049])
    # a3_opt = np.array([-0.00052564, -0.00022404 , 0.00136287])
    # a4_opt = np.array([ 5.22510032e-04, -7.84409037e-05 ,-1.45819627e-03])
    # a5_opt = np.array([-0.00056578 ,-0.0011998  ,0.00068137])
    # a6_opt = np.array([-0.00010426 , 0.00097613 , 0.00064957])
    # b = 0.06308935
    a = np.stack((a1_opt, a2_opt, a3_opt, a4_opt, a5_opt, a6_opt))
    x = testXn[:,n,:]

    # 对于每一组进行每个元素相乘然后求和
    c = sum(np.sum(a * (np.squeeze(x)), axis=1))-b
    print("the %d test sample dis===%f"%(n,c))

for i in range(testXn.shape[1]):
    testx(i)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
#
# class Relevant(nn.Module):
#     def __init__(self,input_size):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(input_size, 128)
#         self.relu = nn.ReLU()
#         self.linear0 = nn.Linear(128, 256)
#         self.linear1 = nn.Linear(256, 128)
#         self.linear2 = nn.Linear(128,9)
#         self.bn = nn.BatchNorm1d(256)
#
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear(x)
#         out = self.relu(x)
#         out = self.linear0(out)
#         out = self.bn(out)
#         out = self.linear1(out)
#
#         out = self.linear2(out)
#         out = out.view(out.shape[0],3,3)
#
#         return out
#
# input_size = 9
# lr_model = Relevant(input_size).cuda()
# X = xyz.transpose(2, 0, 1)[:,3:,:]
# y = xyz.transpose(2, 0, 1)[:,:3,:]
#
# tx = test.transpose(2, 0, 1)[:,3:,:]
# ty = test.transpose(2, 0, 1)[:,:3,:]
#
# # 转换为 PyTorch 的 Tensor
# X_tensor = torch.tensor(X, dtype=torch.float32,requires_grad=True).cuda()
# y_tensor = torch.tensor(y, dtype=torch.float32,requires_grad=True).cuda()
#
# tx_tensor = torch.tensor(tx, dtype=torch.float32,requires_grad=True).cuda()
# ty_tensor = torch.tensor(ty, dtype=torch.float32,requires_grad=True).cuda()
#
# # 将数据包装成 Dataset
# dataset = TensorDataset(X_tensor, y_tensor)
# teda = TensorDataset(tx_tensor,ty_tensor)
#
# # 使用 DataLoader 划分批次
# batch_size = 16
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# test_dataloader = DataLoader(teda,batch_size=4,shuffle=True)
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# # optimizer = optim.SGD(lr_model.parameters(), lr=0.001)
# optimizer = optim.AdamW(lr_model.parameters(), lr=0.001,weight_decay=0.01,betas=(0.9, 0.999))
#
# def train():
#     # 训练模型
#     num_epochs = 100000
#     best_loss = 100000
#     for epoch in range(num_epochs):
#         lr_model.train()
#         for batch_X, batch_y in dataloader:
#             # 前向传播
#             outputs = lr_model(batch_X)
#
#             # 计算损失
#             loss = criterion(outputs, batch_y)
#             # 反向传播和优化
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         lr_model.eval()
#         lossd = 0.
#         for tbatch_X, tbatch_y in test_dataloader:
#             out = lr_model(tbatch_X)
#             lossd += criterion(out, tbatch_y)
#         # 打印训练信息
#         if 0.2*loss.item() + lossd.item()/len(test_dataloader) < best_loss:
#             best_loss = 0.2*loss.item() + lossd.item()/len(test_dataloader)
#             print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')
#             print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {lossd.item()/len(test_dataloader):.4f}')
#             torch.save(lr_model,"relevant_model2.pth")
#
#
# def testre():
#     model = torch.load("relevant_model.pth")
#     model.eval()
#     for tbatch_X, tbatch_y in test_dataloader:
#         out = model(tbatch_X)
#         loss = criterion(out, tbatch_y)
#         # print("out==",out)
#         # print("label==", tbatch_y)
#         print("loss===",loss.item())
#
# # testre()
# train()