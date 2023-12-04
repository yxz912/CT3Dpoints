import os
import pandas as pd
import shutil
import cv2
import send2trash
import numpy as np
import math
from tqdm import tqdm

def copy_file(source_path, destination_path):
    shutil.copytree(source_path, destination_path)

def print_folder_files(folder_path):
    file_count = 0
    files = os.listdir(folder_path)
    for file_name in files:
        file_count += 1
        print(file_name)
    print("文件数量：", file_count)

def print_excel_column(file_path, column_name):
    df = pd.read_excel(file_path)
    column_data = df[column_name]
    non_nan_data = column_data.dropna().tolist()
    for i,data in enumerate(non_nan_data):
        #print(data)
        if data!=i+1:
            print(data)
    print(len(non_nan_data))

def excel_folder_files(folder_path,file_path,column_name,save_path,file1d):
    file_count = 0
    ceshi=0
    files = os.listdir(folder_path)
    files1=os.listdir(file1d)
    df = pd.read_excel(file_path)
    column_data = df[column_name]
    non_nan_data = column_data.dropna().tolist()
    for data in non_nan_data:
        for file_name in files:
            if file_name == data:
                file_count += 1
                # 替换为要移动的文件的路径
                #source_path = folder_path+file_name
                # 替换为目标文件夹的路径
                #destination_path = save_path+file_name
                #copy_file(source_path, destination_path)
                break
        if file_count == ceshi:
            for file_name1 in files1:
                if file_name1==data :
                    #print("exists===",data)
                    file_count += 1
                    # 替换为的文件的路径
                    source_path =  file1d + file_name1
                    # 替换为目标文件夹的路径
                    destination_path = save_path + file_name1
                    #copy_file(source_path, destination_path)
                    break
        if file_count -1 == ceshi:
            ceshi=file_count
        else:
            print("not exists===",data)
            ceshi=file_count
    print("文件数量：", file_count)


def excel_to_save_floder():
    # 替换为你要打印的文件夹的路径
    folder_path = "/media/yxz/Elements（含已标点）/无托槽（附件不影响）/"
    # print_folder_files(folder_path)
    fild="/media/yxz/Elements（含已标点）/无托槽（附件不影响）/已标过点/"
    # 替换为你要打印的Excel文件的路径
    file_path = "/home/yxz/progress/CT3Dpoints/data/三维坐标表格.xlsx"
    # # 替换为你要打印的列名称
    column_name = "姓名"
    # print_excel_column(file_path, column_name)
    save_path="/media/yxz/新加卷/AAA研究生资料/3DCTpoint/data/"

    excel_folder_files(folder_path,file_path,column_name,save_path,fild)


#excel_to_save_floder()
#print_excel_column("/home/yxz/progress/CT3Dpoints/data/三维坐标表格.xlx","序号")

def chuli_ce():
    # 一级文件夹路径
    folder_path = '/media/yxz/Elements/Image/'

    # 遍历所有二级文件夹
    for dir_name in os.listdir(folder_path):
        dir_pathd = os.path.join(folder_path, dir_name)
        # 获取文件夹中所有文件的名称列表
        files = os.listdir(dir_pathd)
        # 按照A-Z顺序排序文件名列表
        dir_path = sorted(files)

        # 遍历当前二级文件夹，找到第一张 PNG 图片
        png_file = None
        for file_name in dir_path:
            if file_name.endswith('.png'):
                png_file = dir_pathd+"/"+file_name
                break

        if png_file is None:
            print(f"No PNG files found in {dir_name}")
            #os.rmdir(dir_path)
        else:
            # 读取 PNG 图片
            img = cv2.imread(png_file)
            # 提取尺寸信息
            height, width, channels = img.shape
            if height!=width:
                print(f"Folder name: {dir_name}, image size: {width} x {height}")
                #shutil.move(png_file, '/media/yxz/Elements/ce/'+file_name)
                send2trash.send2trash(png_file)
            # else:
            #     print(f"Folder name: {dir_name}, image size: {width} x {height}")

def traverse_and_compe_images(folder_path,savepath):
    # 获取文件夹中所有文件的名称列表
    files = os.listdir(folder_path)

    # 按照A-Z顺序排序文件名列表
    sorted_files = sorted(files)
    chas=[]
    top_three_indices=[]
    # 遍历排序后的文件名列表，读取前后两张图片进行像素差计算和打印
    for i in range(len(sorted_files)-1):
        img1 = cv2.imread(os.path.join(folder_path, sorted_files[i]))
        img2 = cv2.imread(os.path.join(folder_path, sorted_files[i+1]))
        if img1.shape == img2.shape:
            # 将图像拆分成各个通道
            b1, g1, r1 = cv2.split(img1)
            b2, g2, r2 = cv2.split(img2)
            # _, br1 = cv2.threshold(r1, 150, 255, cv2.THRESH_BINARY)
            # _, br2 = cv2.threshold(r2, 150, 255, cv2.THRESH_BINARY)
            # cv2.imshow("j1",br1)
            # cv2.imshow("j2",br2)
            # cv2.waitKey()
            # 对R通道进行相减操作
            # diff = abs(br1-br2)
            # cha = np.sum(diff)

            # 进行模板匹配
            result = cv2.matchTemplate(r1, r2, cv2.TM_CCOEFF_NORMED)
            # 获取相似度（取最大值）
            cha = cv2.minMaxLoc(result)[1]

            chas.append(cha)
            #print(i,"=====",cha)
        else:
            print(i,sorted_files[i])
            print(img2.shape)
            top_three_indices.append(i)
    # 使用sorted数对列表进行排序
    sorted_nums = sorted(chas, reverse=False)  ##Ture取da
    # 获取最大的三个数
    top_three = sorted_nums[:4]
    # 获取最大的三个数的索引
    top_three_indice = [chas.index(num) for num in top_three if num<0.4]
    top_three_indices = top_three_indice+top_three_indices
    print("最大的三个数:", top_three)
    print("它们的索引:", top_three_indices)

    if len(top_three_indices)<2:
        x=1
        while(1):
            top_three_indice = [chas.index(num) for num in top_three if num < 0.4+0.1*x]
            top_three_indices = top_three_indice + top_three_indices
            if len(top_three_indices)<2:
                x=x+1
            else:
                break

    sorted_max_indexs = sorted(top_three_indices, reverse=False)  ##Ture取da
    k=0
    for j,file in enumerate(sorted_files):
        os.makedirs(savepath + str(k) + "/", exist_ok=True)
        if j not in sorted_max_indexs:
            shutil.copy(os.path.join(folder_path, file),savepath+str(k)+"/")
        else:
            k=k+1
            print(k)


def create_folders(folder_path,savepath):
    files = os.listdir(folder_path)
    sorted_files = sorted(files)

    current_folder = None
    current_num = None

    for file in sorted_files:
        if file.endswith('.png'):
            num = file[-9:-4]
            #print(num)
            if current_folder is None or int(num) != current_num + 1:
                # 创建新的文件夹
                current_folder = os.path.join(savepath, num)
                os.makedirs(current_folder, exist_ok=True)

            current_num = int(num)
            # 将图片移动到对应的文件夹中
            shutil.copytree(os.path.join(folder_path, file), current_folder)

# folder_path = '/media/yxz/Elements/Image/安焓溪（王）牵引(OUT)/'  # 文件夹路径
# save_path = '/media/yxz/Elements/ce/'
# create_folders(folder_path,save_path)

def copy_images(source_folder, destination_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_folder, file)
                shutil.copy2(source_file, destination_file)

# source_folder = '/media/yxz/Elements/ce/'  # 源文件夹路径
# destination_folder = '/media/yxz/Elements/Image/安焓溪（王）牵引(OUT)/'  # 目标文件夹路径
# copy_images(source_folder, destination_folder)

 # dicom文件目录
# dicom_dir = "/media/yxz/Elements/Image/"
# path = "/media/yxz/Elements/ce/"
# if os.path.exists(path):
#     shutil.rmtree(path)
# for j in tqdm(os.listdir(dicom_dir)):
#     r_path = os.path.join(dicom_dir, j)
#     traverse_and_compe_images(r_path,path+j+"/")


def check(folder_path):
    # 使用os.listdir获取文件夹下的所有文件和文件夹
    all_items = sorted(os.listdir(folder_path))
    # 遍历所有_items并统计文件数量
    if len(all_items) == 1:
        print("zhiyou yige", folder_path)
    for item in all_items:
       if item=='1' or item == 'a' or item=='01':
            # 使用os.path.join构建完整路径
            item_path = os.path.join(folder_path, item)
            jj=len(os.listdir(item_path))
            # 打印文件数量
            #print(f"文件夹{item_path}中共有 {jj} 个文件。")
            if jj<=100:
                print(f"文件夹{item_path}中共有 {jj} 个文件。")
            #     #send2trash.send2trash(item_path)
            #
            #     # 获取文件夹中所有文件的名称列表
            #     files = os.listdir(item_path)
            #     # 按照A-Z顺序排序文件名列表
            #     sorted_files = sorted(files)
            #     chas = []
            #     for i in range(len(sorted_files) - 1):
            #         img1 = cv2.imread(os.path.join(item_path, sorted_files[i]))
            #         img2 = cv2.imread(os.path.join(item_path, sorted_files[i + 1]))
            #         if img1.shape == img2.shape:
            #             # 将图像拆分成各个通道
            #             b1, g1, r1 = cv2.split(img1)
            #             b2, g2, r2 = cv2.split(img2)
            #             result = cv2.matchTemplate(r1, r2, cv2.TM_CCOEFF_NORMED)
            #             cha = cv2.minMaxLoc(result)[1]
            #             if cha<0.7:
            #                 print(i,"=====",cha)
            #                 chas.append(i)
            #     sorted_max_indexs = sorted(chas, reverse=False)  ##Ture取da
            #     print(sorted_max_indexs)
                # k = 0
                # for j, file in enumerate(sorted_files):
                #     os.makedirs(folder_path+ "/0"+ str(k) + "/" , exist_ok=True)
                #     if j not in sorted_max_indexs:
                #         shutil.move(os.path.join(item_path, file), folder_path+ "/0"+ str(k) + "/")
                #     else:
                #         k = k + 1
                # send2trash.send2trash(item_path)

# dicom_dir = "/media/yxz/Elements/ce/"
# for j in os.listdir(dicom_dir):
#     r_path = os.path.join(dicom_dir, j)
#     check(r_path)

def pic_check_move(dicom_dir):
    for j in sorted(os.listdir(dicom_dir)):
        r_path = os.path.join(dicom_dir, j)
        all_items = sorted(os.listdir(r_path))
        despath = os.path.join("/media/yxz/Elements/train/images/", j)
        os.makedirs(despath, exist_ok=True)
        # if "1" in all_items or "a" in all_items:
        #     print("use 1/a")
        # elif "01" in all_items:
        #     print("use 01")
        # else:
        #     print("use no one",r_path)
        for item in all_items:
            if item == "1" or item =="a":
                foldpath=os.path.join(r_path,item)
                size=sorted(os.listdir(foldpath))

                for i in size:
                    filepath = os.path.join(foldpath, i)
                    if filepath.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                        shutil.copy2(filepath, os.path.join(despath,i))
                break
            elif item == "01":
                foldpath = os.path.join(r_path, item)
                size = sorted(os.listdir(foldpath))

                for i in size:
                    filepath = os.path.join(foldpath, i)
                    if filepath.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                        shutil.copy2(filepath, os.path.join(despath,i))
                break

def get_label(name,data_label):
    result = data_label[data_label['姓名'] == name].index.tolist()
    cc = []
    x=[]
    y=[]
    z=[]
    if len(result) == 1:
        row_index = result[0]
        for k in range(23):
            x_value = data_label.loc[row_index+k, 'x']
            y_value = data_label.loc[row_index+k, 'y']
            z_value = data_label.loc[row_index+k, 'z']
            if pd.isnull(x_value) or pd.isnull(y_value) or pd.isnull(z_value):
                cc.append(name)
            else:
                x.append(x_value)
                y.append(y_value)
                z.append(abs(z_value))
    else:
        print(f"找不到 {name}")
    return max(x),max(y),max(z)

data_label = pd.read_excel("/home/yxz/progress/CT3Dpoints/data/三维坐标表格.xlsx")
folder_train = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/train/images/"
folder_test = "/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/val/images/"

train_list = sorted(os.listdir(folder_train))
test_list = sorted(os.listdir(folder_test))
cc=[]
# for i in train_list:
#     cc += get_label(i,data_label)
# for j in test_list:
#     cc += get_label(j,data_label)

# cf = set(cc)
# print(sorted(cf))
# print(len(cf))
mx=0
my=0
mz=0
for i in train_list:
    x,y,z = get_label(i,data_label)
    mx = x if x>mx else mx
    my = y if y>my else my
    mz = abs(z) if abs(z)>mz else mz

for j in test_list:
    x,y,z = get_label(j,data_label)
    mx = x if x > mx else mx
    my = y if y > my else my
    mz = abs(z) if abs(z) > mz else mz

print(mx,my,mz)


