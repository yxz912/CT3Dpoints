import os
import shutil
from send2trash import send2trash

def process_folder(folder_path):
    # 获取当前文件夹中的子文件夹列表
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # 遍历所有子文件夹
    for subfolder in subfolders:
        # 获取子文件夹中的子文件夹个数
        subfolder_count = len([f for f in os.scandir(subfolder) if f.is_dir()])

        # 判断子文件夹个数是否小于5
        if subfolder_count < 5:
            # 将整个子文件夹移动到回收站
            send2trash(subfolder)
        else:
            # 对子文件夹进行递归处理
            process_folder(subfolder)
folder_path = '/home/yxz/progress/CT3Dpoints/results'
process_folder(folder_path)