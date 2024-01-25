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
import cv2
import torchvision.transforms as transforms
from torch.distributions.normal import Normal

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
    #print(k,name)
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


class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data
        msk=torch.from_numpy(mask)
        #return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask).permute(2, 0, 1)
        #return torch.from_numpy(image).permute(2, 0, 1), msk
        return torch.from_numpy(image), msk

class myResize:
    def __init__(self, size_h=512, size_w=512):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        if (image.shape[1] !=self.size_w) or (image.shape[2] != self.size_h):
            # mask[...,0] = mask[...,0]*(self.size_w/image.shape[1])
            # mask[...,1] = mask[...,1]*(self.size_h/image.shape[2])
            #mask[..., 2] = mask[..., 2] * (self.size_h / image.shape[0])
            image = TF.resize(image, [self.size_h, self.size_w])
        return image, mask


class myRandomEqualize_multichannel:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self,data):
        image, mask = data
        # 随机决定是否进行均衡化
        if np.random.rand() > self.p:
            return image,mask
        # 获取图像的通道数
        num_channels = image.shape[0]

        # 对每个通道进行直方图均衡化
        equalized_channels = [cv2.equalizeHist(image[i, :, :].astype(np.uint8)) for i in range(num_channels) ]

        # 将均衡化后的通道重新组合成多通道图像
        equalized_image = np.stack(equalized_channels, axis=0)

        return equalized_image.astype(np.float32),mask

class myRandomHorizontalFlip:
    def __init__(self, p=0.5,input_size_w=512):
        self.p = p
        self.input_size_w=input_size_w
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            flipped_x = self.input_size_w - mask[..., 0]
            msk = torch.cat((flipped_x.unsqueeze(-1), mask[..., 1:]), dim=-1)
            return TF.hflip(image),msk
        else: return image, mask

class myRandomVerticalFlip:
    def __init__(self, p=0.5,input_size_h=512):
        self.p = p
        self.input_size_h=input_size_h
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            mask[..., 1] = self.input_size_h - mask[..., 1]
            return TF.vflip(image),mask
        else: return image, mask

class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle_range= degree
        self.p = p
    def __call__(self, data):
        image, point = data
        if random.random() < self.p:
            angled = random.uniform(self.angle_range[0], self.angle_range[1])
            # 定义旋转角度（以弧度为单位）
            angle=torch.deg2rad(torch.tensor([angled]))
            # 提取前两个维度，即 x 和 y 坐标
            xy = point[..., :2]
            # 计算旋转后的新坐标
            x_new = xy[..., 0] * torch.cos(angle) - xy[..., 1] * torch.sin(angle)
            y_new = xy[..., 0] * torch.sin(angle) + xy[..., 1] * torch.cos(angle)

            # 将旋转后的坐标保存回原始张量
            point[..., 0] = x_new
            point[..., 1] = y_new
            return TF.rotate(image,angled), point
        else: return image, point

def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )

def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode = config.mode,
            factor = config.factor,
            patience = config.patience,
            threshold = config.threshold,
            threshold_mode = config.threshold_mode,
            cooldown = config.cooldown,
            min_lr = config.min_lr,
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


class EuclideanLoss(nn.Module):
    def __init__(self,config):
        super(EuclideanLoss, self).__init__()
        self.setting_config=config

    def forward(self, output, target,leg,l_dynamic):
        #loss=torch.sqrt(((output - target)**2).sum())
        eg=(output-target)**2
        egd=torch.sqrt(torch.sum(eg,dim=(2,3)))
        # eg = abs(output - target)
        # egd=torch.sum(eg,dim=(2,3))
        loss=egd.sum() / (self.setting_config.num_classes)
        if leg==None:
            return loss
        else:
            return loss +  (self.setting_config.l2_lambda * 0.2 * l_dynamic * leg).item()

class Deepeucloss(nn.Module):
    def __init__(self,config):
        super(Deepeucloss,self).__init__()
        self.euc=EuclideanLoss(config)
        self.config=config

    def forward(self,gt_pre,out,target,leg,l_dynamic):
        outloss=self.euc(out,target,leg,l_dynamic)
        kl =0.0
        if self.config.gauss:
            gt_loss = 0.1 * self.euc(gt_pre[1][1], target, None, 0) + 0.2 * self.euc(gt_pre[2][1], target, None, 0)
            for batch in range(out.shape[0]):
                for point in range(out.shape[1]):
                    mean1 = torch.squeeze(out[batch,point,:,:])
                    meand1 = torch.squeeze(gt_pre[1][1][batch,point,:,:])
                    meand2 = torch.squeeze(gt_pre[2][1][batch, point, :, :])
                    mean2 = torch.squeeze(target[batch,point,:,:])
                    var1 = torch.squeeze(gt_pre[2][0][batch,point,:,:])
                    var2 = torch.tensor([2., 2., 2.]).cuda()
                    #kl += self.calculate_KL_divergence(mean1.cpu().detach().numpy(),diag_matrix1.cpu().detach().numpy(),mean2.cpu().detach().numpy(),diag_matrix2.cpu().detach().numpy())
                    # 构建正态分布对象
                    model_dist = Normal(loc=mean1, scale=var1)
                    model_dist1 = Normal(loc=meand1, scale=var1)
                    model_dist2 = Normal(loc=meand2, scale=var1)
                    true_dist = Normal(loc=mean2, scale=var2)
                    # 计算 KL 散度
                    kl += torch.distributions.kl.kl_divergence(model_dist, true_dist).sum() + 0.2*torch.distributions.kl.kl_divergence(model_dist1, true_dist).sum()+0.2*torch.distributions.kl.kl_divergence(model_dist2, true_dist).sum()
            #print(" ",(1./(self.euc(out,target,None,None)+gt_loss)))
            return outloss + gt_loss + (gt_pre[0].item() * self.config.l2_lambda * l_dynamic) + (1./(1.2*(self.euc(out,target,None,0)+gt_loss)))*kl
        else:
            gt_loss=0.0
            for i,pre in enumerate(gt_pre):
                if i>0:
                    gt_loss += 0.1*(i+1)*self.euc(gt_pre[i][1],target,None,0)
            return outloss + gt_loss + (gt_pre[0].item() * self.config.l2_lambda * l_dynamic)

    def calculate_KL_divergence(self,mean1, cov1, mean2, cov2):
        cov2_inv = np.linalg.inv(cov2)
        kl_divergence = 0.5 * (np.trace(np.dot(cov2_inv, cov1)) +
                               np.dot(np.dot((mean2 - mean1), cov2_inv), (mean2 - mean1)) -
                               len(mean1) + np.log(np.linalg.det(cov2) / np.linalg.det(cov1)))
        if math.isinf(kl_divergence):
            return 1000
        else:
            return kl_divergence


class mmld_Loss(nn.Module):
    def __init__(self,config):
        super(mmld_Loss, self).__init__()
        self.config=config

    def forward(self,gt_pre,out,target,leg,l_dynamic):
        loss = self.nofloss(out,target)
        loss += 0.1*self.nofloss(gt_pre[1][1],target)
        loss += 0.2*self.nofloss(gt_pre[2][1],target)

        if leg==None:
            return loss
        else:
            return loss +  (self.config.l2_lambda * 0.2 * l_dynamic * leg).item() +(gt_pre[0].item() * self.config.l2_lambda * l_dynamic)

    def nofloss(self,out,target):
        eg = (out - target) ** 2
        egd = torch.sqrt(torch.sum(eg, dim=(2, 3)))
        loss = 0.0
        # loss = egd.sum() / (self.setting_config.num_classes)
        for i, ls in enumerate(egd):
            for j, bat in enumerate(ls):
                if torch.min(target[i, j]) >= 0:
                    loss += bat
        return loss