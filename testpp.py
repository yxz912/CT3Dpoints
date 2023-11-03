import pandas as pd
import numpy as np
import os
import cv2
import torch

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

def normalize(img,mean,std):
    img_normalized = (img - mean) / std
    img_normalized = ((img_normalized - np.min(img_normalized))
                      / (np.max(img_normalized) - np.min(img_normalized)))*255.
    return img_normalized

#img = cv2.imread("/media/yxz/Elements/train/images/曹剑（沈）正颌/CT.1.3.46.670589.33.1.63816808395857726800001.5597244408443114737.png")
# 将图像的数据类型转换为浮点型
# 生成一个全零的 3x3x1 数组

img = np.zeros((2, 2, 3))
print(img)

img = img.astype(np.float32)

img=normalize(img,(0.5,0.4,0.3),(0.4,0.2,0.1))
# img=torch.from_numpy(img).permute(2,0,1)
print(img)