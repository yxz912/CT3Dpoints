import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import sys
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

def get_minsize(dicom_dir):
    k=float('inf')
    for j in sorted(os.listdir(dicom_dir)):
        r_path = os.path.join(dicom_dir, j)
        size_png = len(sorted(os.listdir(r_path)))
        # print(size_png)
        # if 300<size_png<400:
        #     print(size_png,r_path)
        if k>size_png:
            k=size_png
            name=r_path
    print(k,name)
    return k

def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger

def log_config_info(config, logger):
    config_dict = config.__dict__    #获取配置对象config中的所有属性,并以字典的形式返回
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':   #遍历config_dict中的键值对,其中键名以_开头的属性不输出
            continue
        else:
            log_info = f'{k}: {v},'   #将键值对拼接成 "key: value" 的格式加入log_info，且末尾有个逗号分割
            logger.info(log_info)

