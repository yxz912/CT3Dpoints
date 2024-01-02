from torchvision import transforms
from utils import *
import math
from datetime import datetime

class setting_config:
    network ='Pointneted'#'Pointneted_plus' #'pointnet3d' #'Pointneted' #'egeunet' #'resnet101'     # 'UNet'   #    #'resnet50'
    data_mmld = True
    pre_net = None#"/home/yxz/progress/CT3Dpoints/results/Pointneted_plus_Wednesday_13_December_2023_17h_07m_39s/checkpoints/best.pth"
    data_path='/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/'
    label_path="/media/yxz/新加卷/teeth_ct_points/CT3Dpoints/三维坐标表格.xlsx"
    pretrained_path = './pre_trained/'
    cfg = [32,64,96,128,256,128,256,128,64,32,16] #[96, 128, 256, 512, 1024, 512, 256, 128, 64, 32,16]  #[64, 96, 128,64,32,256,512,256,128,64,32]
    label_num=[0,1,10,11,14,19]  #[2,3,8,9,15,16,17,18,21,22] #[0,1,2,3,8,9,10,11,14,15,16,17,18,19,21,22]
    num_classes = len(label_num)
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
    l2_lambda = 0.00 # 设置 L2 局部正则化权重
    gpu_id = '0'
    batch_size = 4
    val_bs = 1
    epochs = 400
    deep_supervision = True
    tailadd = True
    threshold = 5
    rand_data = False
    mean_common = False
    horizontal_flip = True
    Dynamic_regularization = False
    freeze = True

    if data_mmld:
        network = 'Pointneted' #'Pointneted_plus_mmld'
        threshold = 4

    if rand_data:
        train_mean=41.68455
        train_std=54.433514
        val_mean=42.785255
        val_std=54.970303
    else:
        if not mean_common:
            #50
            # train_mean = [14.954868, 14.954868, 14.954868, 14.289434, 14.289434, 14.289434, 14.1125, 14.1125, 14.1125, 14.056176, 14.056176, 14.056176, 13.785328, 13.785328, 13.785328, 13.722053, 13.722053, 13.722053, 13.750269, 13.750269, 13.750269, 13.865391, 13.865391, 13.865391, 13.599483, 13.599483, 13.599483, 13.461443, 13.461443, 13.461443, 13.129756, 13.129756, 13.129756, 13.074999, 13.074999, 13.074999, 12.753448, 12.753448, 12.753448, 12.493439, 12.493439, 12.493439, 12.677932, 12.677932, 12.677932, 13.07967, 13.07967, 13.07967, 13.719422]
            # train_std = [22.87497, 22.87497, 22.87497, 22.112495, 22.112495, 22.112495, 21.797848, 21.797848, 21.797848, 21.781454, 21.781454, 21.781454, 21.28348, 21.28348, 21.28348, 21.095448, 21.095448, 21.095448, 20.92146, 20.92146, 20.92146, 20.896275, 20.896275, 20.896275, 20.506004, 20.506004, 20.506004, 20.212786, 20.212786, 20.212786, 19.626486, 19.626486, 19.626486, 19.707392, 19.707392, 19.707392, 18.69551, 18.69551, 18.69551, 18.23203, 18.23203, 18.23203, 18.210863, 18.210863, 18.210863, 18.354801, 18.354801, 18.354801, 18.88994]
            # val_mean = [14.826912, 14.826912, 14.826912, 14.212075, 14.212075, 14.212075, 13.595315, 13.595315, 13.595315, 13.550468, 13.550468, 13.550468, 13.305538, 13.305538, 13.305538, 13.118886, 13.118886, 13.118886, 12.883329, 12.883329, 12.883329, 12.984672, 12.984672, 12.984672, 12.604348, 12.604348, 12.604348, 12.2998, 12.2998, 12.2998, 11.946927, 11.946927, 11.946927, 11.675448, 11.675448, 11.675448, 11.444169, 11.444169, 11.444169, 11.2411375, 11.2411375, 11.2411375, 11.48266, 11.48266, 11.48266, 11.8663025, 11.8663025, 11.8663025, 12.958656]
            # val_std = [21.361225, 21.361225, 21.361225, 20.89243, 20.89243, 20.89243, 20.298468, 20.298468, 20.298468, 19.97749, 19.97749, 19.97749, 19.718876, 19.718876, 19.718876, 19.648207, 19.648207, 19.648207, 19.188372, 19.188372, 19.188372, 19.072523, 19.072523, 19.072523, 18.565912, 18.565912, 18.565912, 18.150368, 18.150368, 18.150368, 17.534283, 17.534283, 17.534283, 16.902555, 16.902555, 16.902555, 16.275156, 16.275156, 16.275156, 15.702714, 15.702714, 15.702714, 15.829111, 15.829111, 15.829111, 16.032715, 16.032715, 16.032715, 17.156271]
            #64
            train_mean = [38.167084, 36.47113, 35.658463, 34.7275, 34.397972, 34.392757, 33.96709, 34.126865, 34.28234,
                          33.729267, 32.7962, 32.064926, 31.681631, 31.124203, 30.844501, 31.361849, 32.73974, 34.215652,
                          36.648277, 38.611458, 40.539837, 42.120747, 43.28669, 44.45494, 45.614132, 46.21989, 47.45023,
                          48.500004, 49.44821, 50.419426, 52.140045, 53.649082, 54.722622, 55.427254, 55.996006, 56.55678,
                          57.03274, 56.78339, 56.16994, 56.052105, 56.084873, 55.73664, 55.51461, 54.455627, 53.19349,
                          51.708523, 50.496777, 48.39798, 46.618237]
            train_std = [52.12329, 51.107494, 50.941055, 50.223755, 50.00628, 50.154346, 49.390217, 49.584064, 49.46689,
                         48.635002, 46.955963, 45.86415, 45.254314, 44.008198, 43.081245, 43.15782, 44.181343, 45.16833,
                         47.433487, 48.60615, 50.44894, 51.635715, 52.542904, 53.638527, 54.554443, 54.860928, 55.634895,
                         56.127644, 56.314377, 56.52658, 57.31575, 57.992775, 58.40813, 58.65052, 58.6726, 58.786034, 59.043797,
                         59.15101, 58.878242, 59.1303, 59.565495, 59.95643, 60.605595, 60.74457, 60.784153, 60.811424, 61.18004,
                         60.75567, 60.8064]

            val_mean = train_mean
            val_std = train_std
            # val_mean = [39.743374, 38.122738, 37.19561, 35.71686, 35.202812, 34.790627, 34.621647, 34.678947, 34.44854,
            #             33.327152, 32.52173, 31.267305, 30.603254, 30.013475, 29.536041, 30.480452, 31.952286, 33.71057,
            #             36.010574, 38.201912, 39.313183, 42.94376, 44.53766, 45.518337, 46.94797, 46.58542, 47.49811, 48.238167,
            #             51.026104, 52.292915, 52.69654, 53.169758, 53.87486, 55.697067, 57.654236, 57.586315, 57.758514,
            #             57.174618, 57.736595, 57.24419, 57.544117, 57.37176, 56.43062, 55.96641, 55.325653, 53.10815, 52.179287,
            #             50.27785, 47.04305]
            # val_std = [53.15056, 52.15993, 51.616505, 50.33102, 50.21509, 50.081203, 50.03935, 49.86299, 49.531567, 48.28274,
            #            46.95426, 45.172287, 43.96861, 42.88531, 41.24616, 42.37183, 43.43441, 44.75589, 46.571957, 48.567627,
            #            49.173023, 51.97025, 53.31617, 54.147503, 55.217754, 54.256313, 54.82137, 55.228252, 57.024765,
            #            57.493202, 57.21687, 57.12449, 57.009945, 58.07164, 58.968643, 58.816093, 58.97068, 58.471672, 58.976105,
            #            58.958794, 59.510704, 60.026344, 60.16703, 60.841858, 61.621956, 61.22611, 61.83866, 62.08808, 61.1031]
        else:
            train_mean = [14.954868, 14.954868, 14.954868, 14.289434, 14.289434, 14.289434, 14.1125, 14.1125, 14.1125,
                          14.056176, 14.056176, 14.056176, 13.785328, 13.785328, 13.785328, 13.722053, 13.722053,
                          13.722053, 13.750269, 13.750269, 13.750269, 13.865391, 13.865391, 13.865391, 13.599483,
                          13.599483, 13.599483, 13.461443, 13.461443, 13.461443, 13.129756, 13.129756, 13.129756,
                          13.074999, 13.074999, 13.074999, 12.753448, 12.753448, 12.753448, 12.493439, 12.493439,
                          12.493439, 12.677932, 12.677932, 12.677932, 13.07967, 13.07967, 13.07967, 13.719422]
            train_std = [22.87497, 22.87497, 22.87497, 22.112495, 22.112495, 22.112495, 21.797848, 21.797848, 21.797848,
                         21.781454, 21.781454, 21.781454, 21.28348, 21.28348, 21.28348, 21.095448, 21.095448, 21.095448,
                         20.92146, 20.92146, 20.92146, 20.896275, 20.896275, 20.896275, 20.506004, 20.506004, 20.506004,
                         20.212786, 20.212786, 20.212786, 19.626486, 19.626486, 19.626486, 19.707392, 19.707392,
                         19.707392, 18.69551, 18.69551, 18.69551, 18.23203, 18.23203, 18.23203, 18.210863, 18.210863,
                         18.210863, 18.354801, 18.354801, 18.354801, 18.88994]
            val_mean = [14.826912, 14.826912, 14.826912, 14.212075, 14.212075, 14.212075, 13.595315, 13.595315,
                        13.595315, 13.550468, 13.550468, 13.550468, 13.305538, 13.305538, 13.305538, 13.118886,
                        13.118886, 13.118886, 12.883329, 12.883329, 12.883329, 12.984672, 12.984672, 12.984672,
                        12.604348, 12.604348, 12.604348, 12.2998, 12.2998, 12.2998, 11.946927, 11.946927, 11.946927,
                        11.675448, 11.675448, 11.675448, 11.444169, 11.444169, 11.444169, 11.2411375, 11.2411375,
                        11.2411375, 11.48266, 11.48266, 11.48266, 11.8663025, 11.8663025, 11.8663025, 12.958656]
            val_std = [21.361225, 21.361225, 21.361225, 20.89243, 20.89243, 20.89243, 20.298468, 20.298468, 20.298468,
                       19.97749, 19.97749, 19.97749, 19.718876, 19.718876, 19.718876, 19.648207, 19.648207, 19.648207,
                       19.188372, 19.188372, 19.188372, 19.072523, 19.072523, 19.072523, 18.565912, 18.565912,
                       18.565912, 18.150368, 18.150368, 18.150368, 17.534283, 17.534283, 17.534283, 16.902555,
                       16.902555, 16.902555, 16.275156, 16.275156, 16.275156, 15.702714, 15.702714, 15.702714,
                       15.829111, 15.829111, 15.829111, 16.032715, 16.032715, 16.032715, 17.156271]


    work_dir = 'results/' + network + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
    print(network + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss'))

    if network=='pointnet3d':
        tensor_size=1
    else:
        tensor_size=0

    train_transformer = transforms.Compose([
       # myRandomEqualize_multichannel(p=0.5),  # 自适应直方图均衡化
        myToTensor(),
        #myRandomHorizontalFlip(p=0.3,input_size_w=input_size_w),
        #myRandomVerticalFlip(p=0.3,input_size_h=input_size_h),
        #myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01  # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9  # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6  # default: 1e-6 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adagrad':
        lr = 0.01  # default: 0.01 – learning rate
        lr_decay = 0  # default: 0 – learning rate decay
        eps = 1e-10  # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001  # default: 1e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0.0001  # default: 0 – weight decay (L2 penalty)
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001  # default: 1e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2  # default: 1e-2 – weight decay coefficient
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'Adamax':
        lr = 2e-3  # default: 2e-3 – learning rate
        betas = (0.9,
                 0.999)  # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'ASGD':
        lr = 0.01  # default: 1e-2 – learning rate
        lambd = 1e-4  # default: 1e-4 – decay term
        alpha = 0.75  # default: 0.75 – power for eta update
        t0 = 1e6  # default: 1e6 – point at which to start averaging
        weight_decay = 0  # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        momentum = 0  # default: 0 – momentum factor
        alpha = 0.99  # default: 0.99 – smoothing constant
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False  # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        etas = (0.5,
                1.2)  # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50)  # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
    elif opt == 'SGD':
        lr = 0.01  # – learning rate
        momentum = 0.9  # default: 0 – momentum factor
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        dampening = 0  # default: 0 – dampening for momentum
        nesterov = True  # default: False – enables Nesterov momentum

    sch = 'CosineAnnealingLR' #ReduceLROnPlateau
    if sch == 'StepLR':
        step_size = epochs // 5  # – Period of learning rate decay.
        gamma = 0.5  # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]  # – List of epoch indices. Must be increasing.
        gamma = 0.1  # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99  # – Multiplicative factor of learning rate decay.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50  # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001  # – Minimum learning rate. Default: 0.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'  # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1  # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10  # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001  # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel'  # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0  # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0  # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08  # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50  # – Number of iterations for the first restart.
        T_mult = 2  # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6  # – Minimum learning rate. Default: 0.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
