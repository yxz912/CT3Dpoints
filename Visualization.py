import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from models.Pointnet_ed import Pointneted_plus,Pointnet_ed,Pointneted
from configs.config_setting import setting_config
import os
import cv2
import math

input_channels=49
model = Pointnet_ed(num_classes=setting_config.num_classes,
                    input_channels=input_channels,
                    cfg=setting_config.cfg,
                    deep_supervision=setting_config.deep_supervision,
                    tailadd=setting_config.tailadd,
                    )
model_path = "/home/yxz/progress/CT3Dpoints/TESTMODEL/best4.pth"
data_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/train/images/"
label_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/三维坐标表格.xlsx"

def test(model_path,data_path,label_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    images_list = sorted(os.listdir(data_path))
    for i,img_path in enumerate(images_list):
        size = len(sorted(os.listdir(data_path+img_path)))
        common = math.ceil(size / 64)
        img = concat_cv(input_channels,data_path+img_path,common)
        img = cv2.resize(img, [setting_config.input_size_h , setting_config.input_size_w])
        img = (img - setting_config.train_mean) / setting_config.train_std
        img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.
        img = np.transpose(img, (2, 0, 1))
        img=torch.from_numpy(img)
        data = torch.unsqueeze(img, 0)
        model.eval()
        model(data.cuda().float())
        break

target_layer = model.tail1
feature_maps = []

# 定义一个钩子函数来获取目标层的特征图
def hook_fn(module, input, output):
    feature_maps.append(output)

def concat_cv(real_input_channels,folder, common=1):
    img_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.png')]
    # 获取图像的宽度和高度
    img = cv2.imdecode(np.fromfile(img_paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    # 创建一个与图像尺寸相同的数组（初始化为零）
    r_concat = np.zeros((height, width, 1), dtype=np.uint8)
    rc = r_concat.copy()
    # print(r_concat.shape)

    jj = 0
    for i, img_path in enumerate(img_paths):
        if jj < real_input_channels:
            if i % common == 0:  ##randc
                jj = jj + 1
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                r = img[:, :, 0]
                r = np.stack((r,) * 1, axis=-1)
                if jj == 1:
                    r_concat = r.copy()
                else:
                    r_concat = np.concatenate((r_concat, r), axis=2)

            elif i == len(img_paths) - 1:
                jj = jj + 1
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                r = img[:, :, 2]
                r = np.stack((r,) * 1, axis=-1)
                r_concat = np.concatenate((r_concat, r), axis=2)
    return r_concat

# 注册钩子函数
hook = target_layer.register_forward_hook(hook_fn)
test(model_path,data_path,label_path)
# 取消注册钩子
hook.remove()

# 取出第一个通道的特征图并绘制热力图
feature_map = torch.squeeze(feature_maps[0][0],dim=1)
plt.figure(figsize=(8, 8))
print(feature_map.shape)
sns.heatmap(feature_map.cpu().detach().numpy(), cmap='viridis')
plt.title('Visualization of Feature Map')
plt.show()
