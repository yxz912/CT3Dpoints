import cv2
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
import shutil
import math
import pandas as pd
import random

class YXZ_datasets(Dataset):
    def __init__(self,path_Data,label_path,config,train=True):
        super(YXZ_datasets, self)
        self.input_channels = config.input_channels
        self.num_cls = config.num_classes
        self.in_h=config.input_size_h
        self.in_w=config.input_size_w
        self.real_input_channels,self.commons_train,self.commons_test = self.get_min_input_channels(path_Data+'train/images/',path_Data + 'val/images/')
        # 读取excel 文件
        self.data_label = pd.read_excel(label_path)
        self.tensize = config.tensor_size
        self.rand_data = config.rand_data
        self.mean_common = config.mean_common
        self.horizontal_flip = config.horizontal_flip
        self.label_num=config.label_num

        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/'))
            self.data=[]
            self.train_size = len(images_list)
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/'+images_list[i]
                mask_name = images_list[i]
                self.data.append([img_path,mask_name])
            if self.rand_data:
                #self.mean, self.std = self.get_mean_std_all(path_Data, train)
                self.mean,self.std=config.train_mean,config.train_std
                print("train_mean=", self.mean)
                print("train_std=", self.std)
            else:
                if len(config.train_mean)!=self.real_input_channels or len(config.train_std)!=self.real_input_channels:
                    self.mean,self.std=self.get_mean_std(path_Data,train)
                    print("train_mean=",self.mean)
                    print("train_std=", self.std)
                else:
                    self.mean,self.std=config.train_mean,config.train_std
            #self.mean_label,self.std_label=self.get_label_standardization(namelist=images_list)
            self.transformer = config.train_transformer
        else:
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            self.val_size = len(images_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_name = images_list[i]
                self.data.append([img_path, mask_name])
            if self.rand_data:
                #self.mean, self.std = self.get_mean_std_all(path_Data, train)
                self.mean, self.std = config.val_mean, config.val_std
                print("val_mean=", self.mean)
                print("val_std=", self.std)
            else:
                if len(config.val_mean) != self.real_input_channels or len(config.val_std) != self.real_input_channels:
                    self.mean, self.std = self.get_mean_std(path_Data,train)
                    print("val_mean=", self.mean)
                    print("val_std=", self.std)
                else:
                    self.mean, self.std = config.val_mean, config.val_std
            #self.mean_label, self.std_label = self.get_label_standardization(namelist=images_list)
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path,msk_path = self.data[indx]
        size=len(sorted(os.listdir(img_path)))
        common=math.ceil(size/self.input_channels)

        img = self.concat_cv(img_path, common)
        msk = self.get_label(msk_path,hf=False)
        if self.horizontal_flip==True and random.random() < 0.5:
            img = cv2.flip(img, 1)
            msk = self.get_label(msk_path,hf=True)
        img= (img - self.mean) / self.std
        img = ((img- np.min(img)) / (np.max(img) - np.min(img))) * 255.
        #img = ((img - np.min(img)) / (np.max(img) - np.min(img)))

        if self.tensize:
            img = cv2.resize(img,(self.in_w, self.in_h))
            img = img.reshape((self.real_input_channels, self.in_w, self.in_h, 1))
        else:
            img = np.transpose(img, (2, 0, 1))

        img,msk = self.transformer((img,msk))
        return img,msk

    def __len__(self):
        return len(self.data)

    def concat_cv(self,folder,common=1):
        img_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.png')]
        # 获取图像的宽度和高度
        height, width = cv2.imdecode(np.fromfile(img_paths[0] ,dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2]

        # 创建一个与图像尺寸相同的数组（初始化为零）
        r_concat = np.zeros((height, width,1), dtype=np.uint8)
        rc = r_concat.copy()
        #print(r_concat.shape)

        remainder = len(img_paths)//common - self.real_input_channels
        randr=0
        if remainder>1:
            listr = list(range(0, remainder*common//2, 1))
            randr = random.choice(listr)

        jj=0
        listc = list(range(0, common, 1))
        randc = random.choice(listc)

        for i,img_path in enumerate(img_paths):
            if i < randr and self.rand_data:
                continue
            if jj<self.real_input_channels:
                if i%common == 0:   ##randc
                    if self.mean_common :
                        rc = np.zeros_like(rc)
                        jkm = 0
                        jj = jj + 1
                        for t in range(common):
                            if i+t < len(img_paths):
                                img = cv2.imdecode(np.fromfile(img_paths[i+t], dtype=np.uint8), cv2.IMREAD_COLOR)
                                jkm+=1
                            else:
                                continue
                            r = img[:, :, 2]
                            r = np.stack((r,) * 1, axis=-1)
                            rc += r
                        rc = rc//jkm
                        if jj==1:
                            r_concat = rc.copy()
                        else:
                            r_concat=np.concatenate((r_concat, rc), axis=2)
                    else:
                        jj=jj+1
                        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                        r = img[:,:,0]
                        r = np.stack((r,) * 1, axis=-1)
                        if jj==1:
                            r_concat = r.copy()
                        else:
                            r_concat=np.concatenate((r_concat, r), axis=2)

                elif i == len(img_paths)-1:
                    jj=jj+1
                    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    r = img[:, :, 2]
                    r = np.stack((r,) * 1, axis=-1)
                    r_concat = np.concatenate((r_concat, r), axis=2)
        return r_concat

    def get_min_input_channels(self,dicom_dir,dic2):
        channels=[]
        commons_train=[]
        commons_test=[]
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
            commons_train.append(common)
        for j in sorted(os.listdir(dic2)):
            r_path = os.path.join(dic2, j)
            size = len(sorted(os.listdir(r_path)))
            common = math.ceil(size / self.input_channels)
            channel = math.ceil(size / common)
            channels.append(channel)
            commons_test.append(common)
        real_input_channels  = min(channels)
        return real_input_channels,commons_train,commons_test

    def get_label(self,name,hf=False):
        result = self.data_label[self.data_label['姓名'] == name].index.tolist()
        array_2d = np.zeros((self.num_cls, 3))
        if len(result) == 1:
            row_index = result[0]
            #if self.horizontal_flip:
            for i,k in enumerate(self.label_num):
                x_value = self.data_label.loc[row_index + k, 'x']
                y_value = self.data_label.loc[row_index + k, 'y']
                z_value = self.data_label.loc[row_index + k, 'z'] + 100
                if hf:
                    # if i%2==0:
                    #     z_value = self.data_label.loc[row_index + k+1, 'z'] + 100
                    # else:
                    #     z_value = self.data_label.loc[row_index + k-1, 'z'] + 100
                    z_value = -1*self.data_label.loc[row_index + k, 'z'] + 100
                array_2d[i] = [x_value, y_value, z_value]
            # else:
            #     for k in range(array_2d.shape[0]):
            #         x_value = self.data_label.loc[row_index+k, 'x']
            #         y_value = self.data_label.loc[row_index+k, 'y']
            #         z_value = self.data_label.loc[row_index+k, 'z'] + 100
            #         array_2d[k]=[x_value,y_value,z_value]
        else:
            print(f"找不到 {name}")
        array_2d=np.expand_dims(array_2d, axis=1)
        return array_2d

    def get_mean_std(self,dir,trian=True):
        imgs_path_list=[]
        if trian:
            print(f"#------Calculate train mean and std ing...------#")
            fold= os.path.join(dir, 'train/images/')
            for k,i in enumerate(sorted(os.listdir(fold))):
                fg=os.path.join(fold,i)
                img_paths = [os.path.join(fg, f) for f in sorted(os.listdir(fg)) if f.endswith('.png')]
                imgst=[]
                img = cv2.imdecode(np.fromfile(img_paths[0] ,dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, 0]
                for kk,timg in enumerate(img_paths):
                    if  kk % self.commons_train[k] == 0 and len(imgst)<self.real_input_channels:
                        if self.mean_common:
                            img = np.zeros_like(img)
                            hh = 0
                            for t in range(self.commons_train[k]):
                                if kk+t < len(img_paths):
                                    img=cv2.imdecode(np.fromfile(img_paths[kk+t], dtype=np.uint8), cv2.IMREAD_COLOR)
                                    img+=img[:,:,0,np.newaxis]
                                    hh+=1
                            imgst.append(img//hh)
                        else:
                            img = cv2.imdecode(np.fromfile(timg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            img = img[:, :, 0, np.newaxis]
                            imgst.append(img)
                imgd=np.concatenate(imgst,axis=2)
                imgs_path_list.append(imgd)
        else:
            print(f"#------Calculate val mean and std ing...------#")
            fold= os.path.join(dir, 'val/images/')
            for k,i in enumerate(sorted(os.listdir(fold))):
                fg=os.path.join(fold,i)
                img_paths = [os.path.join(fg, f) for f in sorted(os.listdir(fg)) if f.endswith('.png')]
                imgst=[]
                img = cv2.imdecode(np.fromfile(img_paths[0], dtype=np.uint8), cv2.IMREAD_COLOR)[:, :, 0]
                for kk,timg in enumerate(img_paths):
                    if  kk % self.commons_test[k] == 0 and len(imgst)<self.real_input_channels:
                        if self.mean_common:
                            img = np.zeros_like(img)
                            hh = 0
                            for t in range(self.commons_test[k]):
                                if kk + t < len(img_paths):
                                    img=cv2.imdecode(np.fromfile(img_paths[kk+t], dtype=np.uint8), cv2.IMREAD_COLOR)
                                    img+=img[:,:,0,np.newaxis]
                                    hh+=1
                            imgst.append(img//hh)
                        else:
                            img = cv2.imdecode(np.fromfile(timg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            img = img[:, :, 0, np.newaxis]
                            imgst.append(img)
                imgd=np.concatenate(imgst,axis=2)
                imgs_path_list.append(imgd)

        img_h, img_w = self.in_h,self.in_w
        means, stdevs = [], []
        img_list = []

        for img in imgs_path_list:
            img = cv2.resize(img, (img_w, img_h))
            img = img[:, :, :, np.newaxis]
            img_list.append(img)

        imgs = np.concatenate(img_list, axis=3)
        imgs = imgs.astype(np.float32)

        for i in range(self.real_input_channels):
            pixels = imgs[:, :, i, :].ravel()  # 拉成一行
            means.append(np.mean(pixels))
            stdevs.append(np.std(pixels))
        return means,stdevs

    def get_mean_std_all(self, dir, trian=True):
        imgs_path_list = []
        if trian:
            print(f"#------Calculate train mean and std ing...------#")
            fold = os.path.join(dir, 'train/images/')
            for k, i in enumerate(sorted(os.listdir(fold))):
                fg = os.path.join(fold, i)
                img_paths = [os.path.join(fg, f) for f in sorted(os.listdir(fg)) if f.endswith('.png')]
                imgst = []
                for kk, timg in enumerate(img_paths):
                    img = cv2.imdecode(np.fromfile(timg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    img = img[:, :, 0, np.newaxis]
                    imgst.append(img)
                imgd = np.concatenate(imgst, axis=2)
                imgs_path_list.append(imgd)
        else:
            print(f"#------Calculate val mean and std ing...------#")
            fold = os.path.join(dir, 'val/images/')
            for k, i in enumerate(sorted(os.listdir(fold))):
                fg = os.path.join(fold, i)
                img_paths = [os.path.join(fg, f) for f in sorted(os.listdir(fg)) if f.endswith('.png')]
                imgst = []
                for kk, timg in enumerate(img_paths):
                    img = cv2.imdecode(np.fromfile(timg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    img = img[:, :, 0, np.newaxis]
                    imgst.append(img)
                imgd = np.concatenate(imgst, axis=2)
                imgs_path_list.append(imgd)

        img_h, img_w = self.in_h, self.in_w
        means, stdevs = [], []
        img_list = []

        for img in imgs_path_list:
            img = cv2.resize(img, (img_w, img_h))
            img_list.append(img)

        imgs = np.concatenate(img_list, axis=2)
        imgs = imgs.astype(np.float32)

        if self.rand_data:
              means = np.mean(imgs.ravel())
              stdevs = np.std(imgs.ravel())
        else:
            for i in range(self.real_input_channels):
                pixels = imgs[:, :, i, :].ravel()  # 拉成一行
                means.append(np.mean(pixels))
                stdevs.append(np.std(pixels))
        return means, stdevs

    def get_label_standardization(self,namelist=[]):
        label_mean=[]
        label_std=[]
        for i in range(self.num_cls):
            num0=[]
            for name in namelist:
                result = self.data_label[self.data_label['姓名'] == name].index.tolist()
                if len(result) == 1:
                    row_index = result[0]
                    num0.append(self.data_label.loc[row_index + i, 'x'])
                    num0.append(self.data_label.loc[row_index + i, 'y'])
                    num0.append(self.data_label.loc[row_index + i, 'z'])
            label_mean.append(np.mean(num0))
            label_std.append(np.std(num0))
        return label_mean,label_std
