import pandas as pd
import numpy as np
# 读取 Excel 文件
data = pd.read_excel('/media/yxz/Elements/三维坐标表格.xlsx')

# 在 "姓名" 列中查找某个名字
name_to_find = 'pw（蔡）正颌'
result = data[data['姓名'] == name_to_find].index.tolist()
print(result)
if len(result) == 1:
    row_index = result[0]
    array_2d = np.zeros((3, 3))
    for i in range(3):
        x_value = data.loc[row_index+i, 'x']
        y_value = data.loc[row_index+i, 'y']
        z_value = data.loc[row_index+i, 'z']
        array_2d[i]=[x_value,y_value,z_value]
    print(array_2d)
else:
    print(f"找不到 {name_to_find}")