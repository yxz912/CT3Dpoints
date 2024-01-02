import os
import numpy as np
from torch.utils.data import Dataset
import nrrd
import torch
import cv2
from configs.config_setting import setting_config

class Molar3D(Dataset):
    def __init__(self, transform=None, phase='train', parent_path=None, data_type="full"):

        self.data_files = []
        self.label_files = []
        self.spacing = []

        cur_path = os.path.join(parent_path, str(phase))
        for file_name in os.listdir(cur_path):
            if file_name.endswith('_volume.nrrd'):
                cur_file_abbr = file_name.split("_volume")[0]

                if data_type == "full":
                    _label = np.load(os.path.join(cur_path, cur_file_abbr + "_label.npy"))
                    if np.any(np.sum(_label, 1) < 0):
                        continue
                if data_type == "mini":
                    _label = np.load(os.path.join(cur_path, cur_file_abbr + "_label.npy"))
                    if np.all(np.sum(_label, 1) > 0):
                        continue

                self.data_files.append(os.path.join(cur_path, cur_file_abbr + "_volume.nrrd"))
                self.label_files.append(os.path.join(cur_path, cur_file_abbr + "_label.npy"))
                self.spacing.append(os.path.join(cur_path, cur_file_abbr + "_spacing.npy"))

        self.transform = transform
        print('the data length is %d, for %s' % (len(self.data_files), phase))

    def __len__(self):
        L = len(self.data_files)
        return L

    def __getitem__(self, index):
        _img, _ = nrrd.read(self.data_files[index])
        _landmark = np.load(self.label_files[index])
        _spacing = np.load(self.spacing[index])
        sample = {'image': _img, 'landmarks': _landmark, 'spacing': _spacing}
        #sample = _img, _landmark
        if self.transform is not None:
            sample = self.transform(sample)
            # sample['landmarks'] = sample['landmarks'] / torch.tensor(4.)
            for ldx, landmark in enumerate(sample['landmarks']):
                if min(landmark) <0:
                    sample['landmarks'][ldx] = torch.tensor([-1.,-1.,-1.])
            sample['landmarks'] = sample['landmarks'].reshape((sample['landmarks'].shape[0],1,3))

            return sample
        else:
            sample = Totensor(sample)
            return sample

    def __str__(self):
        pass

def Totensor(sample):
    img = np.array(sample['image']).astype(np.float32)
    img = cv2.resize(img,[setting_config.input_size_h , setting_config.input_size_w])
    # 抽取比例
    scale_factor = 128 / 256
    # 调整图像的通道数目
    img = img[:, :, ::int(1 / scale_factor)]
    img /= 255.
    img = np.transpose(img, (2, 0, 1))
    imgd = torch.from_numpy(img)
    #mk = sample[1].astype(np.float32)/10.
    mark = torch.unsqueeze(torch.from_numpy(sample['landmarks'].astype(np.float32)),2)
    sample = imgd,mark

    return sample

