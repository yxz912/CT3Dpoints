import cv2
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import shutil
import math
import pandas as pd

class YXZ_datasets(Dataset):
    def __init__(self,path_Data,label_path,config,train=True):
        super(YXZ_datasets, self)
        self.input_channels = config.input_channels
        self.num_cls = config.num_classes
        self.real_input_channels = self.get_min_input_channels(path_Data+'train/images/',path_Data + 'val/images/')
        # 读取xcel 文件
        self.data_abel = pd.read_excel(label_path)

        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            self.data=[]
            for i in range(len(images_list)):
                img_path = path_Data + 'train/imgags/'+images_list[i]
                mask_name = images_list[i]
                self.data.append([img_path,mask_name])
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/imgags/' + images_list[i]
                mask_name = images_list[i]
                self.data.append([img_path, mask_name])
            self.transformer = config.train_transformer

    def __getitem__(self, indx):
        img_path,msk_path = self.data[indx]
        size=len(sorted(os.listdir(img_path)))
        common=math.ceil(size/self.input_channels)
        img=self.concat_cv(img_path,common)

    def __len__(self):
        return len(self.data)

    def concat_cv(self,folder,common=1):
        img_paths = [os.pathjoin(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.png')]
        # 获取图像的宽度和高度
        height, width = cv2.imread(img_paths[0]).shape[:2]
        # 创建一个与图像尺寸相同的数组（初始化为零）
        r_concat = np.zeros((height, width,1), dtype=np.uint8)
        #print(r_concat.shape)
        jj=0
        for i,img_path in enumerate(img_paths):
            if i%common==0 and jj<self.real_input_channels:
                jj=jj+1
                if i==0:
                    img = cv2.imread(img_path)
                    r = img[:, :, 2]
                    r = np.stack((r,) * 1, axis=-1)
                    r_concat=r.copy()
                    continue
                img = cv2.imread(img_path)
                r = img[:, :, 2]
                r = np.stack((r,) * 1, axis=-1)
                r_concat=np.concatenate((r_concat, r), axis=2)
        # 转换为tensor
        img_tensor = torch.from_numpy(r_concat).permute(2,0,1)
        #print(img_tensor.shape)
        return img_tensor

    def get_min_input_channels(self,dicom_dir,dic2):
        channels=[]
        for j in sorted(os.listdir(dicom_dir)):
            r_path = os.path.join(dicom_dir, j)
            size = len(sorted(os.listdir(r_path)))
           # common1 = math.ceil(size / self.input_channels)
            # common2 = int(size / self.input_channels)
            # if common2!=1:
            #     channel1 = math.ceil(size/common1)
            #     channel2 = math.ceil(size/common2)
            #     channel = channel1 if channel1>channel2 else channel2
            # else:
            #     channel = math.ceil(size / common1)
            common = math.ceil(size / self.input_channels)
            channel = math.ceil(size / common)
            channels.append(channel)
            channels.append(channel)
        for j in sorted(os.listdir(dic2)):
            r_path = os.path.join(dic2, j)
            size = len(sorted(os.listdir(r_path)))
            common = math.ceil(size / self.input_channels)
            channel = math.ceil(size / common)
            channels.append(channel)
        real_input_channels  = min(channels)
        return real_input_channels

    def get_label(self,name):
        result = self.data_label[self.data_label['姓名'] == name].index.tolist()
        array_2d = np.zeros((self.num_cls,3))
        if len(result) == 1:
            row_index = result[0]
            for i in range(self.num_cls):
                x_value = self.data_label.loc[row_index+i, 'x']
                y_value = self.data_label.loc[row_index+i, 'y']
                z_value = self.data_label.loc[row_index+i, 'z']
                array_2d[i]=[x_value,y_value,z_value]
        else:
            print(f"找不到 {name}")
