import os
import SimpleITK
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import pydicom
import zipfile


def convert_from_dicom_to_jpg(img, low_window, high_window, save_path):
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一化
    newimg = (newimg * 255).astype('uint8')  # 将像素值扩展到[0,255]
    if len(img.shape)==2:
        stacked_img = np.stack((newimg,) * 1, axis=-1)
    else:
        stacked_img = newimg.copy()
    print(stacked_img.shape)
    #cv2.imwrite(save_path, stacked_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #cv2.imwrite(save_path, newimg, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def find_dcm_files(root_folder):
    dcm_files = []
    for folder_path, _, files in os.walk(root_folder):
        for file_name in files:
            if file_name.endswith('.dcm'):
                dcm_files.append(os.path.join(folder_path, file_name))
    # if len(dcm_files)==0:
    #     for folder_path, _, files in os.walk(root_folder):
    #         for file_name in files:
    #             file_path = os.path.join(folder_path, file_name)
    #             # 判断文件是否为.zip文件
    #             if file_name.endswith('.zip'):
    #                 # 打开.zip文件
    #                 with zipfile.ZipFile(file_path, 'r') as zip_ref:
    #                     # 解压所有文件到当前文件夹
    #                     zip_ref.extractall(folder_path)
    #     for folder_path, _, files in os.walk(root_folder):
    #         for file_name in files:
    #             if file_name.endswith('.dcm'):
    #                 dcm_files.append(os.path.join(folder_path, file_name))
    return dcm_files


if __name__ == '__main__':
    # dicom文件目录
    dicom_dir = "/media/yxz/新加卷/AAA研究生资料/3DCTpoint/data/"
    path = "/media/yxz/Elements/Imaged/"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for j in tqdm(os.listdir(dicom_dir)):
        r_path = os.path.join(dicom_dir, j)
        # 获取所有的DCM文件路径列表
        dcm_files = find_dcm_files(r_path)
        if len(dcm_files)==0:
            continue
        if os.path.exists(path+j):
            shutil.rmtree(path+j)
        print(j)
        os.makedirs(path + j)
        for i in dcm_files:
            dcm_image_path = i
            file_name = os.path.basename(i)
            name, _ = os.path.splitext(file_name)
            output_jpg_path = os.path.join(path+j, name + '.png')
            try:
                ds_array = SimpleITK.ReadImage(dcm_image_path)  # 读取dicom文件的相关信息
            except Exception as e:
                # 捕获异常并打印错误消息
                print("读取 DICOM 文件时出错:", str(e))
                continue
            img_array = SimpleITK.GetArrayFromImage(ds_array)  # 获取array
            # SimpleITK读取的图像数据的坐标顺序为zyx，即从多少张切片到单张切片的宽和高，此处我们读取单张，因此img_array的shape
            # 类似于 （1，height，width）的形式
            shape = img_array.shape
            if len(shape)==4:
                #print(shape,output_jpg_path)
                img_array = np.reshape(img_array, (shape[1], shape[2],shape[3]))
                continue
            else:
                img_array = np.reshape(img_array, (shape[1], shape[2]))  # 获取array中的height和width

            high = np.max(img_array)
            low = np.min(img_array)
            convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)