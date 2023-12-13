import torch
from models.Pointnet_ed import Pointneted_plus
from configs.config_setting import setting_config
import os
import cv2
import numpy as np
import math
import pandas as pd

def test_model(model,data):
    model.eval()
    with torch.no_grad():
        count = 0
        loss = 0.0
        val_images, val_labels = data
        if setting_config.deep_supervision:
            pre, outputs = model(val_images.cuda().float())
        else:
            outputs = model(val_images.cuda().float())  # eval model only have last output layer
        eg = (outputs - val_labels.cuda().float()) ** 2
        # eg=torch.sum(eg,dim=(1,2,3))
        # count += (eg <= config.threshold).sum().item()
        eg = torch.sqrt(torch.sum(eg, dim=(2, 3)))
        # for k in range(config.val_bs):
        #     coun = (eg[k] <= config.threshold).sum().item()
        #     print(coun,eg[k])
        #     if coun==3:
        #         count+=1
        #print("eg==",eg)
        count = (eg <= setting_config.threshold).sum().item()
        return count

def test(model_path,data_path,label_path):
    input_channels=49
    model = Pointneted_plus(num_classes=setting_config.num_classes,
                            input_channels=input_channels,
                            cfg=setting_config.cfg,
                            deep_supervision=setting_config.deep_supervision,
                            tailadd=setting_config.tailadd,
                            )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    images_list = sorted(os.listdir(data_path))
    cou=0
    for i,img_path in enumerate(images_list):
        size = len(sorted(os.listdir(data_path+img_path)))
        common = math.ceil(size / 64)
        img = concat_cv(input_channels,data_path+img_path,common)
        msk = get_label(label_path, img_path)
        img = (img - setting_config.train_mean) / setting_config.train_std
        img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.
        img = np.transpose(img, (2, 0, 1))
        msk = torch.from_numpy(msk)
        img=torch.from_numpy(img)
        data = torch.unsqueeze(img, 0),torch.unsqueeze(msk,0)
        coun=test_model(model,data)
        cou+= coun
        print(("name=%s,right_point=%d")%(img_path,coun))
    print("test_acc=",cou/(len(images_list)*setting_config.num_classes))

def concat_cv(real_input_channels,folder, common=1):
    img_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.png')]
    # 获取图像的宽度和高度
    height, width = cv2.imdecode(np.fromfile(img_paths[0], dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]

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
    array_2d=np.expand_dims(array_2d, axis=1)
    return array_2d

model_path = "/home/yxz/progress/CT3Dpoints/TESTMODEL/best.pth" #"/home/yxz/progress/CT3Dpoints/results/Pointneted_plus_Wednesday_13_December_2023_17h_07m_39s/checkpoints/best.pth"
data_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/val/images/"
label_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/三维坐标表格.xlsx"

test(model_path,data_path,label_path)