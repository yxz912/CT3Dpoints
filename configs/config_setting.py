from torchvision import transforms
from utils import *

from datetime import datetime

class setting_config:
    network='Pointnet2d'
    data_path='/media/yxz/Elements/'
    label_path='/media/yxz/Elements/三维坐标表格.xlsx'
    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 64  ##期望最大输入
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 8
    epochs = 100
    deep_supervision = False

    work_dir = 'results/' + network + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    train_transformer = transforms.Compose([
        # myNormalize(datasets, train=True),
        # myToTensor(),
        # myRandomHorizontalFlip(p=0.5),
        # myRandomVerticalFlip(p=0.5),
        # myRandomRotation(p=0.5, degree=[0, 360]),
        # myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        # myNormalize(datasets, train=False),
        # myToTensor(),
        # myResize(input_size_h, input_size_w)
    ])

