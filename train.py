import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import YXZ_datasets
from datasets.data3d import Molar3D
from tensorboardX import SummaryWriter
from engine import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import logging
import torch.optim as optim
from models.resnet import resnet34, resnet101,resnet50,resnet152
from models.Pointnet_ed import Pointneted_plus,Pointneted_plus_mmld,Pointneted,Pointnet_ed,Pointneted_gaus
from models.unet import UNet
from models.egeunet import EGEUNet
from models.Pointnet3d import pointnet3d
import transforms as tr

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
    logging.basicConfig(filename=log_dir+"/train.info.log", level=logging.INFO)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)  # 通过设置随机种子，可以使得随机数产生的结果可重现，即在每次运行代码时得到相同的随机序列。这在涉及到随机性的任务中很有用，例如数据的随机划分、权重的初始化等。
    torch.cuda.empty_cache()  # 用于清空GPU缓存

    print('#----------Preparing dataset----------#')
    if config.data_mmld:
        train_transform = transforms.Compose([
            tr.RandomCrop(),  # zoom and random crop for data augumentation
            #tr.LandmarkProposal(shrink=args.shrink, anchors=args.anchors),  # generate the anchor proposal
            tr.Normalize(),
            tr.ToTensor(),
        ])
        val_transform = transforms.Compose([
            tr.CenterCrop(),  # zoom and random crop for data augumentation
            # tr.LandmarkProposal(shrink=args.shrink, anchors=args.anchors),  # generate the anchor proposal
            tr.Normalize(),
            tr.ToTensor(),
        ])
        train_dataset = Molar3D(None,'train','/home/yxz/data/mmld_code/mmld_code/mmld_dataset','all')
        train_loader = DataLoader(train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=config.num_workers)
        val_dataset = Molar3D(None, 'val', '/home/yxz/data/mmld_code/mmld_code/mmld_dataset', 'all')
        val_loader = DataLoader(val_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
        train_dataset.real_input_channels=config.input_channels
        val_dataset.val_size=100
        train_dataset.train_size=458

    else:
        train_dataset = YXZ_datasets(config.data_path,config.label_path, config, train=True)
        logging.info("train--mean-->%s",np.array(train_dataset.mean))
        logging.info("train--std-->%s",np.array(train_dataset.std))

        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=config.num_workers)
        val_dataset = YXZ_datasets(config.data_path, config.label_path,config, train=False)
        logging.info("val--mean-->%s",np.array(val_dataset.mean))
        logging.info("val--std-->%s",np.array(val_dataset.std))
        logging.info("real_input_channels-->%d",val_dataset.real_input_channels)
        val_loader = DataLoader(val_dataset,
                                batch_size=config.val_bs,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)

    print('#----------Prepareing Model----------#')
    if config.network == 'resnet50':
        model = resnet50(input_channels=train_dataset.real_input_channels,
                     num_classes=config.num_classes,
                     )
    elif config.network == 'resnet101':
        model = resnet50(input_channels=train_dataset.real_input_channels,
                     num_classes=config.num_classes,
                     )
    elif config.network == 'UNet':
        model = UNet(n_channels=train_dataset.real_input_channels,
                         n_classes=config.num_classes,
                         )
    elif config.network == 'egeunet':
        model = EGEUNet(num_classes=config.num_classes,
                        input_channels=train_dataset.real_input_channels,
                        c_list=[128,256,512,1024,512,128],
                         bridge=True,
                         gt_ds=True,
                        deep_supervision=config.deep_supervision
        )
    elif config.network == 'pointnet3d':
        model = pointnet3d(n_classes=config.num_classes,
                           n_channels=train_dataset.real_input_channels
        )
    elif config.network == 'Pointneted_plus':
        model = Pointneted_plus(num_classes=config.num_classes,
                           input_channels=train_dataset.real_input_channels,
                           cfg=config.cfg,
                           deep_supervision=config.deep_supervision,
                           tailadd = config.tailadd,
                           )
    elif config.network == 'Pointneted_plus_mmld':
        model = Pointneted_plus_mmld(num_classes=config.num_classes,
                                input_channels=train_dataset.real_input_channels,
                                cfg=config.cfg,
                                deep_supervision=config.deep_supervision,
                                tailadd=config.tailadd,
                                )
    elif  config.network == 'Pointneted':
        model = Pointneted(num_classes=config.num_classes,
                                input_channels=train_dataset.real_input_channels,
                                cfg=config.cfg,
                                deep_supervision=config.deep_supervision,
                                tailadd=config.tailadd,

                                )
    elif config.network == 'Pointnet_ed':
        model = Pointnet_ed(num_classes=config.num_classes,
                           input_channels=train_dataset.real_input_channels,
                           cfg=config.cfg,
                           deep_supervision=config.deep_supervision,
                           tailadd = config.tailadd,
                           )

    elif config.network == 'Pointneted_gaus':
        model = Pointneted_gaus(num_classes=config.num_classes,
                           input_channels=train_dataset.real_input_channels,
                           cfg=config.cfg,
                           deep_supervision=config.deep_supervision,
                           tailadd = config.tailadd,
                           )
    else:
        raise Exception('network in not right!')
    model = model.cuda()

    print('#----------Prepareing loss, opt, sch and amp----------#')
    if config.deep_supervision:
        loss_function = Deepeucloss(config)
    else:
        loss_function = EuclideanLoss(config)
    # if config.data_mmld:
    #     loss_function = mmld_Loss(config)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    l_dynamic=1.0
    if config.pre_net:
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(config.pre_net, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        l_dynamic = checkpoint['l_dynamic']

    print('#----------train and test start...----------#')
    for i in range(2):
        if i==0:
            vv,mec_val,mec_train= simple_train_val(config, model, train_loader, val_loader, optimizer, loss_function, logging, scheduler,
                                  val_dataset.val_size, train_dataset.train_size, l_dynamic)
            # 使用 matplotlib 绘制迭代图
            plt.subplot(1, 2, 1)
            plt.legend()
            plt.plot(range(len(vv)), vv,color='red', label="NO Relevance")
            plt.subplot(1, 2, 2)
            plt.plot(range(len(mec_val)), mec_val,color='red', label="NOT Relevance Adjust")
            plt.legend()
        if i==1:
            model = Pointneted(num_classes=config.num_classes,
                           input_channels=train_dataset.real_input_channels,
                           cfg=config.cfg,
                           deep_supervision=config.deep_supervision,
                           tailadd = config.tailadd,
                           ).cuda()
            config.gauss = False
            l_dynamic = 1.0
            optimizer = get_optimizer(config, model)
            scheduler = get_scheduler(config, optimizer)
            vv,mec_val,mec_train = simple_train_val(config, model, train_loader, val_loader, optimizer, loss_function, logging, scheduler,
                                  val_dataset.val_size, train_dataset.train_size, l_dynamic)
            # 使用 matplotlib 绘制迭代图
            plt.subplot(1, 2, 1)
            plt.legend()
            plt.plot(range(len(vv)), vv,color='blue', label="Relevance")
            plt.subplot(1, 2, 2)
            plt.plot(range(len(mec_val)), mec_val, color='blue',label="Relevance Adjust")
            plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    # 添加标题和标签
    plt.title(f"Test Accuracy--{str(config.network) + '--' + str(config.freeze)}")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    os.mkdir(config.work_dir + "plt/")
    plt.savefig(config.work_dir + "plt/" + config.network + '.png')
    # 显示图形
    plt.show()