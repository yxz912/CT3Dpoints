import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import YXZ_datasets
from tensorboardX import SummaryWriter
from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")  ##警告过滤

if __name__ == '__main__':
    config = setting_config
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')
    log_config_info(config, logger)  # 配置记录

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)  # 通过设置随机种子，可以使得随机数产生的结果可重现，即在每次运行代码时得到相同的随机序列。这在涉及到随机性的任务中很有用，例如数据的随机划分、权重的初始化等。
    torch.cuda.empty_cache()  # 用于清空GPU缓存

    print('#----------Preparing dataset----------#')
    train_dataset = YXZ_datasets(config.data_path,config.label_path, config, train=True)
    print(train_dataset.real_input_channels)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = YXZ_datasets(config.data_path, config.label_path,config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

