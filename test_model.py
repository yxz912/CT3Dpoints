import torch
from models.Pointnet_ed import Pointneted_plus,Pointnet_ed
from configs.config_setting import setting_config
import os
import cv2
import numpy as np
import math
import pandas as pd
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        ed = torch.sqrt(eg)
        disx = torch.unsqueeze(ed[:,:,:,0].cpu(),dim=-1)
        disy = torch.unsqueeze(ed[:, :, :, 1].cpu(),dim=-1)
        disz = torch.unsqueeze(ed[:, :, :, 2].cpu(),dim=-1)

        # eg=torch.sum(eg,dim=(1,2,3))
        # count += (eg <= config.threshold).sum().item()
        eg = torch.sqrt(torch.sum(eg, dim=(2, 3)))
        count = (eg <= setting_config.threshold).sum().item()

        outputs_np = outputs.cpu().numpy()[0, :, 0, :]  # 变成 [6, 3]
        val_labels_np = val_labels.cpu().numpy()[0, :, 0, :]  # 变成 [6, 3]
        return count,outputs_np,val_labels_np,disx,disy,disz,eg

def test(model_path,data_path,label_path):
    input_channels=49
    model = Pointnet_ed(num_classes=setting_config.num_classes,
                            input_channels=input_channels,
                            cfg=setting_config.cfg,
                            deep_supervision=setting_config.deep_supervision,
                            tailadd=setting_config.tailadd,
                            )
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    for name, param in model.named_parameters():
        print(name)
    images_list = sorted(os.listdir(data_path))
    cou=0
    outputs=[]
    labelsd=[]
    x=[]
    y=[]
    z=[]
    egd = torch.zeros([1,6])
    for i,img_path in enumerate(images_list):
        size = len(sorted(os.listdir(data_path+img_path)))
        common = math.ceil(size / 64)
        img = concat_cv(input_channels,data_path+img_path,common)
        img = cv2.resize(img, [setting_config.input_size_h , setting_config.input_size_w])
        msk = get_label(label_path, img_path)
        img = (img - setting_config.train_mean) / setting_config.train_std
        img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255.
        img = np.transpose(img, (2, 0, 1))
        msk = torch.from_numpy(msk)
        img=torch.from_numpy(img)
        data = torch.unsqueeze(img, 0),torch.unsqueeze(msk,0)
        coun,out,lab,disx,disy,disz,eg=test_model(model,data)
        cou+= coun
        outputs.append(out.tolist().copy())
        labelsd.append(lab.tolist()[:])
        x.append(torch.mean(disx))
        y.append(torch.mean(disy))
        z.append(torch.mean(disz))
        egd += eg.cpu()
        print(("name=%s,right_point=%d")%(img_path,coun))
    print("test_acc=",cou/(len(images_list)*setting_config.num_classes))
    # 创建一个新的图形和 3D 轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制输出坐标点，使用红色
    ax.scatter(np.array(outputs)[:, 0], np.array(outputs)[:, 1], np.array(outputs)[:, 2], color='r', label='Outputs')
    # 绘制标签坐标点，使用蓝色
    ax.scatter(np.array(labelsd)[:, 0], np.array(labelsd)[:, 1], np.array(labelsd)[:, 2], color='b', label='Labels')
    # 添加图例
    ax.legend()
    # 添加坐标轴标签
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    # 显示图表
    plt.show()

    plt.plot(range(len(x)), x, label="x")
    plt.plot(range(len(y)), y, label="y")
    plt.plot(range(len(z)), z, label="z")
    # 添加曲线标签
    plt.legend()
    # 添加标题和标签
    plt.title(f"Distance--x,y,z")
    plt.xlabel('sample')
    plt.ylabel('distance')
    # 显示图形
    plt.show()

    data =np.squeeze(np.array(egd/len(images_list),dtype=float))
    print(data)
    # 创建索引
    indices = np.arange(len(data))
    # 绘制矩形图
    plt.bar(indices, data)

    # 自定义 x 轴标签
    labels = ['ANS', 'B1', 'U1', 'Me', 'N', 'S']
    plt.xticks(indices, labels)

    # 添加坐标轴标签
    plt.xlabel('Category')
    plt.ylabel('Average Distance Value')
    # 显示图表
    plt.show()


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

def get_label(label_path,name):
    data_label = pd.read_excel(label_path)
    result = data_label[data_label['姓名'] == name].index.tolist()
    array_2d = np.zeros((setting_config.num_classes, 3))
    if len(result) == 1:
        row_index = result[0]
        for i,k in enumerate(setting_config.label_num):
            x_value =data_label.loc[row_index + k, 'x']
            y_value = data_label.loc[row_index + k, 'y']
            z_value = data_label.loc[row_index + k, 'z'] + 128  ####128
            array_2d[i] = [x_value, y_value, z_value]
    else:
        print(f"找不到 {name}")
    array_2d=np.expand_dims(array_2d, axis=1)
    return array_2d

model_path = "/home/yxz/progress/CT3Dpoints/TESTMODEL/best4.pth"
data_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/val/images/"
label_path = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/三维坐标表格.xlsx"

test(model_path,data_path,label_path)
