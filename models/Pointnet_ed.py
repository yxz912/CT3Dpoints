import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

class LayerNorm(nn.Module):
    r"""
    实现了层内归一化(Layer Normalization),这个LayerNorm类就提供了一个通用的层内归一化实现,既支持NCHW也支持NHWC两种格式。
    对于输入特征矩阵（如全连接层或卷积层的输出），计算每个样本在特征维度上的均值和标准差。
    对每个样本进行特征维度的归一化。对于每个特征维度的每个样本，通过减去均值并除以标准差，将其映射到均值为0、标准差为1的分布。
    在归一化后，通过一个可学习的缩放因子和偏移量，对归一化后的特征进行线性变换。这个缩放因子和偏移量可以根据模型的需求进行学习，使得网络可以自适应地调整归一化后的特征。
    将线性变换后的特征作为归一化的最终输出，供下一层的处理使用。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # print(x.shape)
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Conv4DLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # 初始化可学习参数
        if self.data_format == "channels_last":
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        elif self.data_format == "channels_first":
            self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1, 1, 1))

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, x.size()[1:], self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(dim=(2, 3, 4), keepdim=True)
            s = (x - u).pow(2).mean(dim=(2, 3, 4), keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight * x + self.bias
            return x

class Deepwisenn(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.dwn = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=2,dilation=2,groups=dim_in),  ##dilation: 膨胀或空洞系数,默认是1;  groups: 分组卷积,设置为dim_in即为depthwise卷积
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.GroupNorm(4, dim_out),
            #nn.Dropout(p=0.5)
        )

        self.dwn4d=nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, padding=2,dilation=2),  ##dilation: 膨胀或空洞系数,默认是1;  groups: 分组卷积,设置为dim_in即为depthwise卷积
            nn.GELU(),
            # nn.Conv3d(dim_in, dim_out, kernel_size=3,stride=1,padding=1),
            nn.GroupNorm(4, dim_out),
            #nn.Dropout(p=0.5)
        )

    def forward(self,x):
        if len(x.shape)==4:
            x = self.dwn(x)
        else:
            x=self.dwn4d(x)
        return x

class Self_Attention_3D(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        x=y=8
        k_size = 3  # 设置卷积核的大小为3。
        pad = (k_size - 1) // 2  # 计算卷积操作的padding大小，以保持输入输出的尺寸一致。

        # self.params_c = nn.Parameter(torch.Tensor(1,dim_in,x,y),
        #                               requires_grad=True)

        self.params_xy = nn.Parameter(torch.Tensor(1, dim_in, x, y),
                                      requires_grad=True)  # p：定义一个可学习的参数 params_xy，它是一个具有大小为 (1, dim_in, x, y) 的四维张量。
        nn.init.ones_(self.params_xy)  # 对参数 params_xy 进行初始化，将其填充为全1的张量。
        # 定义一个卷积层 conv_xy，其中包含多个操作：分组卷积操作、GELU激活函数以及再次的卷积操作。
        self.conv_xy = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=k_size, padding=pad, groups=dim_in),
                                     nn.GELU(), nn.Conv2d(dim_in, dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, dim_in, x),
                                      requires_grad=True)  # 定义一个可学习的参数 params_zx，它是一个具有大小为 (1, 1, dim_in, x) 的四维张量。
        nn.init.ones_(self.params_zx)  # 对参数 params_zx 进行初始化，将其填充为全1的张量。后类似---
        self.conv_zx = nn.Sequential(nn.Conv1d(dim_in, dim_in, kernel_size=k_size, padding=pad, groups=dim_in),
                                     nn.GELU(), nn.Conv1d(dim_in, dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(dim_in, dim_in, kernel_size=k_size, padding=pad, groups=dim_in),
                                     nn.GELU(), nn.Conv1d(dim_in, dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in)
        )  ##定义一个深度卷积层 dw，其中包含多个操作：1x1卷积、GELU激活函数以及3x3分组卷积操作。深度可分离卷积的参数数量较少，从而减少了模型的复杂度。

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')  # 定义一个标准化层norm1，用于对输入进行归一化处理。
        self.norm2 = LayerNorm(dim_in+dim_out, eps=1e-6, data_format='channels_first')  # 定义一个标准化层 norm2，用于对特征图进行归一化处理。

        self.dwn = Deepwisenn(dim_in,dim_out)

    def forward(self,x):
        px = x
        # ----------xy----------#
        params_xy = self.params_xy  # 获取参数 params_xy (1,C,X,Y)->(1,C,W,H)
        x1 = x * self.conv_xy(F.interpolate(params_xy, size=x.shape[2:4], mode='bilinear',
                                             align_corners=True))  ##上采样用的双线性插值，变尺寸与x1大小一致，对 x1 进行卷积操作，并与 params_xy 进行元素逐元素相乘。
        # ----------zx----------#
        x = x.permute(0, 3, 1, 2)  # (N,C,W,H)->(N,H,C,W)   对 x2 进行维度重排，将通道维度移动到前面。将tensor调整为Conv1d层需要的输入格式
        params_zx = self.params_zx  ###########(1,1,C,X)->(1,1,C,W)->(1,C,W)->(1,1,C,W)
        x2 = x * self.conv_zx(
            F.interpolate(params_zx, size=x.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(
            0)  # squeeze,unsqueeze去掉并加回batch维，align_corners=True表示在采样时保留角点像素的位置。
        x = x.permute(0, 2, 3, 1)  # 对 x 进行维度重排，将通道维度移动到最后面。(N,H,C,W)->(N,C,W,H)
        x2 = x2.permute(0,2,3,1)
        # ----------zy----------#
        x = x.permute(0, 2, 1, 3)  # 对 x 进行维度重排，交换通道维度与横向维度。(N,C,W,H)->(N,W,C,H)
        params_zy = self.params_zy  ###(1,1,C,Y)->(1,1,C,H)->(1,C,H)->(1,1,C,H)
        x3 = x * self.conv_zy(
            F.interpolate(params_zy, size=x.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x = x.permute(0, 2, 1, 3)  # 对 x 进行维度重排，交换通道维度与横向维度。(N,C,W,H)->(N,C,W,H)
        x3 = x3.permute(0,2,1,3)
        # ----------dw----------#
        px = self.dw(px)  # 对 x 进行深度卷积操作

        dx = torch.add(torch.add(torch.add(x1, x2), x3), px)  # add
        dx = self.norm1(dx)
        dx = self.dwn(dx)
        dx = torch.cat([x, dx], dim=1)
        dx = self.norm2(dx)

        return dx

class Self_Attention_4D(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        x = y = z= 8
        k_size = 3  # 设置卷积核的大小为3。
        pad = (k_size - 1) // 2  # 计算卷积操作的padding大小，以保持输入输出的尺寸一致。

        self.params_xyz = nn.Parameter(torch.Tensor(1, dim_in, x, y, z),
                                      requires_grad=True)
        nn.init.ones_(self.params_xyz)  # 对参数 params_xyz 进行初始化，将其填充为全1的张量。
        # 定义一个卷积层 conv_xyz，其中包含多个操作：分组卷积操作、GELU激活函数以及再次的卷积操作。
        self.conv_xyz = nn.Sequential(nn.Conv3d(dim_in, dim_in, kernel_size=k_size, padding=pad, groups=dim_in),
                                     nn.GELU(), nn.Conv3d(dim_in, dim_in, 1))

        self.params_cxy = nn.Parameter(torch.Tensor(1, 1, dim_in, x,y),
                                      requires_grad=True)
        nn.init.ones_(self.params_cxy)  # 对参数 params_zxy 进行初始化，将其填充为全1的张量。后类似---
        self.conv_cxy = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=k_size, padding=pad, groups=dim_in),
                                     nn.GELU(), nn.Conv2d(dim_in, dim_in, 1))

        self.params_cyz = nn.Parameter(torch.Tensor(1, 1, dim_in, y,z), requires_grad=True)
        nn.init.ones_(self.params_cyz)
        self.conv_cyz = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=k_size, padding=pad, groups=dim_in),
                                     nn.GELU(), nn.Conv2d(dim_in, dim_in, 1))

        self.params_cxz = nn.Parameter(torch.Tensor(1, 1, dim_in, x, z), requires_grad=True)
        nn.init.ones_(self.params_cxz)
        self.conv_cxz = nn.Sequential(nn.Conv2d(dim_in, dim_in, kernel_size=k_size, padding=pad, groups=dim_in),
                                      nn.GELU(), nn.Conv2d(dim_in, dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv3d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
        )  ##定义一个深度卷积层 dw，其中包含多个操作：1x1卷积、GELU激活函数以及3x3分组卷积操作。深度可分离卷积的参数数量较少，从而减少了模型的复杂度。

        self.norm1 = Conv4DLayerNorm(dim_in, eps=1e-6, data_format='channels_first')  # 定义一个标准化层norm1，用于对输入进行归一化处理。
        self.norm2 = Conv4DLayerNorm(dim_in+dim_out, eps=1e-6, data_format='channels_first')  # 定义一个标准化层 norm2，用于对特征图进行归一化处理。

        self.dwn = Deepwisenn(dim_in, dim_out)

    def forward(self,x):
        px = x
        #--------------xyz---------------#
        params_xyz = self.params_xyz
        x1 = x * self.conv_xyz(F.interpolate(params_xyz,size=x.shape[2:5], mode='trilinear',align_corners=True))
        #---------------cxy---------------#
        params_cxy = self.params_cxy
        x = x.permute(0,4,1,2,3)
        x2 = x * self.conv_cxy(F.interpolate(params_cxy,size=x.shape[2:5], mode='trilinear',align_corners=True).squeeze(0)).unsqueeze(0)
        x = x.permute(0,2,3,4,1)
        x2 = x2.permute(0, 2, 3, 4, 1)
        #--------------cyz---------------#
        params_cyz = self.params_cyz
        x = x.permute(0,2,1,3,4)
        x3 = x * self.conv_cyz(F.interpolate(params_cyz,size=x.shape[2:5], mode='trilinear',align_corners=True).squeeze(0)).unsqueeze(0)
        x = x.permute(0,2,1,3,4)
        x3 = x3.permute(0, 2, 1, 3, 4)
        #-------------cxz----------------#
        params_cxz = self.params_cxz
        x = x.permute(0,3,1,2,4)
        x4 = x * self.conv_cxz(F.interpolate(params_cxz,size=x.shape[2:5], mode='trilinear',align_corners=True).squeeze(0)).unsqueeze(0)
        x = x.permute(0,2,3,1,4)
        x4 = x4.permute(0, 2, 3, 1, 4)
        #------------dw------------------#
        px = self.dw(px)

        dx = torch.add(torch.add(torch.add(torch.add(x1, x2), x3), x4),px)  # add
        dx = self.norm1(dx)
        dx = self.dwn(dx)
        dx = torch.cat([x,dx],dim=1)
        dx = self.norm2(dx)

        return dx

class Ascension_3t4D(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.norma = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norml = Conv4DLayerNorm(dim_out, eps=1e-6, data_format='channels_first')
        self.covnd1 = Deepwisenn(dim_in,dim_in)
        self.covnd2 = Deepwisenn(dim_out,dim_out)
        self.dim_out = dim_out

    def forward(self,x):
        x = self.covnd1(x)
        x = self.norma(x)
        tx = x.view(x.shape[0],self.dim_out,x.shape[2],x.shape[3],-1)
        tx = self.covnd2(tx)
        tx = self.norml(tx)

        return tx

class lower_4t3D(nn.Module):
    def __init__(self,dim_in,dim_out,channel=1):
        super().__init__()
        self.channel = channel
        self.norma = LayerNorm(dim_out, eps=1e-6, data_format='channels_first')
        self.norml = Conv4DLayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.covnd1 = Deepwisenn(dim_in, dim_in)
        self.covnd2 =  nn.Sequential( nn.Conv2d(dim_in*channel, dim_in*channel, kernel_size=3, padding=2,dilation=2,groups=dim_in*channel),
                        nn.GELU(),
                        nn.Conv2d(dim_in*channel, dim_out, kernel_size=1))

    def forward(self,x):
        x = self.covnd1(x)
        x = self.norml(x)
        tx = x.view(x.shape[0],x.shape[1]*x.shape[4],x.shape[2],x.shape[3])
        tx = self.covnd2(tx)
        tx = self.norma(tx)

        return tx


# 定义 RBF 模块
'''
RBFLayer用作卷积神经网络的特征提取层。
RBFLayer实现了基于径向基函数(RBF)的卷积操作。它首先使用普通的卷积层提取特征图。然后应用径向基函数来生成新的特征表示。
RBFLayer会:
使用conv_rbf卷积层对输入进行卷积操作,得到out_channels * num_centers 个特征图
将特征图reshape为out_channels个通道,每个通道有num_centers个特征图
对每个通道的num_centers个特征图,计算与learnable参数rbf_weights的欧式距离(径向基函数)
对每个通道沿着num_centers这个维度求和,生成out_channels个新特征图
所以RBFLayer学习了一个新的特征表示,可以用来代替普通的卷积层,作为CNN网络的特征提取模块。它通常可以学习更加判别性的特征,提高模型区分样本的能力。
RBFLayer可以很好地作为卷积神经网络特征提取层使用,取代部分普通卷积层,以学习更有判别性的特征。
'''

class RBFLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_centers, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_centers = num_centers

        self.conv_rbf = nn.Conv2d(in_channels, out_channels*num_centers, kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=False)

        self.rbf_weights = nn.Parameter(torch.randn(out_channels, num_centers))

    def forward(self, x):
        batch_size, _, height, width = x.size()

        x_rbf = self.conv_rbf(x)  # Apply convolution

        x_rbf = x_rbf.view(batch_size, self.out_channels, self.num_centers, height, width)  # Reshape

        rbf_output = (x_rbf - self.rbf_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) ** 2  # Calculate RBF output

        rbf_output = torch.sum(rbf_output, dim=2)  # Sum along the num_centers dimension

        return rbf_output


class Pointneted(nn.Module):
    def __init__(self,num_classes=3,input_channels=64,cfg=[96,128,256,512,1024,512,256,128,64,32],deep_supervision=True):
        super().__init__()

        self.deepsuper = deep_supervision
        self.num_classes = num_classes
        #self.weight = nn.Parameter(torch.ones(4))

        self.layer1 = nn.Sequential(nn.Conv2d(input_channels, cfg[0], kernel_size=3,stride=2, padding=1),
                                    nn.GELU(),
                                    Self_Attention_3D(cfg[0], cfg[1]),
                                    nn.Conv2d(cfg[1]+cfg[0], cfg[1], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[1],eps=1e-6, data_format='channels_first'),
                                    # nn.Dropout(p=0.5)
        )

        self.layer2 = nn.Sequential( nn.Conv2d(cfg[1], cfg[2],  kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d( cfg[2],cfg[3],  kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[3], eps=1e-6, data_format='channels_first')
        )

        self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[5],  kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[5], cfg[6],  kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[6], eps=1e-6, data_format='channels_first')
        )

        self.layer4 = nn.Sequential(nn.Conv2d(cfg[6], cfg[5],  kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d( cfg[5],cfg[4],  kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[4], cfg[6],  kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[6], eps=1e-6, data_format='channels_first'),
                                    # nn.Dropout(p=0.5)
        )

        self.layer5 = nn.Sequential(
                                    #Self_Attention_3D(cfg[6],cfg[7]),
                                    Ascension_3t4D(cfg[6],cfg[8]),
                                    Self_Attention_4D(cfg[8],cfg[8]),
                                    Deepwisenn(cfg[7], cfg[8]),
                                    Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
        )

        self.addx = nn.Sequential(
                        nn.Conv2d(cfg[3]+cfg[1], cfg[3], kernel_size=3, stride=2, padding=1),
                        nn.GELU(),
                        nn.Conv2d(cfg[3] , cfg[2], kernel_size=3, stride=1, padding=1),
                        LayerNorm(cfg[2], eps=1e-6, data_format='channels_first'),
                        # nn.Dropout(p=0.5)
        )

        self.tail1 = nn.Sequential(  nn.GroupNorm(4, cfg[8]),
                                    nn.Conv3d(cfg[8], cfg[7], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    Conv4DLayerNorm(cfg[7], eps=1e-6, data_format='channels_first'),
                                    nn.Conv3d(cfg[7], cfg[8], kernel_size=3, stride=2, padding=1),
                                    lower_4t3D(cfg[8],cfg[9],2),
                                     LayerNorm(cfg[9], eps=1e-6, data_format='channels_first'),
        )

        self.tail2 = nn.Sequential( nn.Conv2d(cfg[9]+cfg[6], cfg[7], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[7], eps=1e-6, data_format='channels_first'),
                                    nn.Conv2d(cfg[7], cfg[9], kernel_size=3, stride=2, padding=1),
                                    nn.Conv2d(cfg[9], num_classes, kernel_size=1),
                                    nn.AdaptiveAvgPool2d((1, 3))
        )

        self.deep1 = nn.Sequential( nn.GroupNorm(4, cfg[1]),
                                    nn.Conv2d(cfg[1], cfg[5], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[5], eps=1e-6, data_format='channels_first'),
                                    nn.Conv2d(cfg[5], cfg[8], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[8], eps=1e-6, data_format='channels_first'),
                                    nn.Conv2d(cfg[8], num_classes, kernel_size=3, stride=2, padding=1),
                                    nn.AdaptiveAvgPool2d((1,3))
        )

        self.deep2 = nn.Sequential( nn.GroupNorm(4, cfg[2]),
                                    nn.Conv2d(cfg[2], cfg[7], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[7], eps=1e-6, data_format='channels_first'),
                                    nn.Conv2d(cfg[7], num_classes, kernel_size=3, stride=2, padding=1),
                                    nn.AdaptiveAvgPool2d((1,3))

        )

        self.up_sample1 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                       nn.Conv2d(288, cfg[1], kernel_size=3, stride=1, padding=1),
        )
        self.up_sample2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(288, cfg[2], kernel_size=3, stride=1, padding=1),
                                        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        '''
        对于nn.Linear层，使用trunc_normal_方法将权重初始化为均值为0，标准差为0.02的截断正态分布，并将偏置初始化为0。
        对于nn.Conv1d层，根据该层的卷积核大小、输出通道数和输入通道数计算fan_out，并使用正态分布方法将权重初始化为均值为0，标准差为sqrt(2.0 / n)的值。
        对于nn.Conv2d层，根据该层的卷积核大小、输出通道数、输入通道数和组数计算fan_out，并使用正态分布方法将权重初始化为均值为0，标准差为sqrt(2.0 / fan_out)的值，并将偏置初始化为0。
        '''
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,x):
        x = self.layer1(x)    ## ,
        x1 = self.layer2(x)    ##,
        x2 = self.layer3(x1)  ##,
        x3 = self.layer4(x2)  ##,
        x4 = torch.cat((x2,x3),dim=1) ##,
        x5 = torch.cat((torch.add(x1,x4),x),dim=1)  ##,
        x5 = self.addx(x5)    ##,
        x6 = self.layer5(x5)  ##,
        x7 = self.tail1(x6)
        x7 = torch.cat((x7,F.max_pool2d(x5,kernel_size=2)),dim=1)
        x8 = self.tail2(x7)

        if self.deepsuper:
            dx1 = self.deep1(torch.add(self.up_sample1(x7),x))
            dx2 = self.deep2(torch.add(self.up_sample2(x7),x5))
            return [[0.2,dx1],[0.4,dx2]],x8
        else:
            return x8


class Pointneted_plus(nn.Module):
    def __init__(self,num_classes=3,input_channels=64,cfg=[96,128,256,512,1024,512,256,128,64,32,16],deep_supervision=True,tailadd=True):
        super().__init__()

        self.deepsuper = deep_supervision
        self.tail_add = tailadd
        self.num_classes = num_classes
        #self.weight = nn.Parameter(torch.ones(4))

        self.head = nn.Sequential(nn.Conv2d(input_channels,cfg[0],kernel_size=3,stride=1,padding=1),
                                    nn.GELU(),
        )

        self.layer1 = nn.Sequential(nn.Conv2d(cfg[0],cfg[0],kernel_size=3,stride=1,padding=1,groups=cfg[0]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[1], eps=1e-6, data_format='channels_first'),
        )

        self.layer2 = nn.Sequential(nn.Conv2d(cfg[1],cfg[1],kernel_size=3,stride=1,padding=1,groups=cfg[1]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[3], eps=1e-6, data_format='channels_first'),
        )

        self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[3], kernel_size=3, stride=1, padding=1,groups=cfg[3]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[4], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[5], cfg[6], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    LayerNorm(cfg[6], eps=1e-6, data_format='channels_first'),
        )

        self.layer4 = nn.Sequential(nn.Conv2d(cfg[6]+cfg[0], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    Deepwisenn(cfg[5],cfg[6]),
        )

        self.A3D1 = nn.Sequential( Self_Attention_3D(cfg[6], cfg[6]),
                                   nn.Conv2d(2*cfg[6],cfg[7],kernel_size=3,stride=1,padding=1),
                                   LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
        )

        self.A3D2 = nn.Sequential( Self_Attention_3D(cfg[7], cfg[7]),
                                   nn.Conv2d(2*cfg[7],cfg[8],kernel_size=3,stride=2,padding=1),
                                   LayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
        )

        self.A3D3 = nn.Sequential(Self_Attention_3D(cfg[8]+cfg[7], cfg[7]),
                                  nn.Conv2d(cfg[8]+2*cfg[7], cfg[7], kernel_size=3, stride=1, padding=1),
                                  LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                  )

        self.A4D1 = nn.Sequential(Self_Attention_4D(cfg[8], cfg[9]),
                                  nn.Conv3d(cfg[8]+cfg[9], cfg[8], kernel_size=3, stride=1, padding=1),
                                  Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  )

        self.A4D2 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  )

        self.A4D3 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  )

        self._3t4_1 = Ascension_3t4D(cfg[7],cfg[8])

        self._3t4_2 = Ascension_3t4D(cfg[8],cfg[10])

        self._3t4_3 = Ascension_3t4D(cfg[7],cfg[10])

        self._4t3_1 = lower_4t3D(cfg[8],cfg[9],2)

        self._4t3_2 = lower_4t3D(cfg[8],cfg[8],4)

        self._4t3_3 = lower_4t3D(cfg[8],cfg[7],8)

        self.head_l1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1)

        self.l1_l2 = nn.Conv2d(cfg[1], cfg[3], kernel_size=3,stride=2,padding=1)

        self.l2_l3 = nn.Conv2d(cfg[3], cfg[6], kernel_size=3,stride=2,padding=1)

        self.wl1 = nn.Conv2d(cfg[3],cfg[8],kernel_size=1,stride=1)

        self.wl2 = nn.Conv2d(cfg[6],cfg[9],kernel_size=1,stride=1)

        self.head_l3 = nn.MaxPool2d(kernel_size=4)

        self.bridge1 = nn.Conv2d(cfg[9],cfg[8],kernel_size=1)

        self.bridge2 = nn.Conv2d(cfg[8],cfg[7],kernel_size=1)

        self.up_sample1 = nn.Conv3d(cfg[9],cfg[10],kernel_size=1)

        self.up_sample2 = nn.Conv3d(cfg[8],cfg[10],kernel_size=1)

        self.tail1 = nn.Sequential(nn.GroupNorm(4, cfg[9]),
                                   nn.Conv2d(cfg[9], num_classes, kernel_size=3, stride=2, padding=1),
                                   nn.AdaptiveAvgPool2d((1,3))
        )

        self.tail2 = nn.Sequential(nn.GroupNorm(4, cfg[8]),
                                   nn.Conv2d(cfg[8], cfg[10], kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(cfg[10], num_classes, kernel_size=3, stride=2, padding=1),
                                   nn.AdaptiveAvgPool2d((1,3))
        )

        self.tail3 = nn.Sequential(nn.GroupNorm(4, cfg[7]),
                                   nn.Conv2d(cfg[7], cfg[8], kernel_size=3, stride=2, padding=1),
                                   nn.GELU(),
                                   nn.Conv2d(cfg[8], cfg[10], kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(cfg[10], num_classes, kernel_size=3, stride=2, padding=1),
                                   nn.AdaptiveAvgPool2d((1,3))
        )

        self.tailadd1 = nn.Conv2d(cfg[7], cfg[8], kernel_size=3, stride=2, padding=1)

        self.tailadd2 = nn.Conv2d(cfg[8], cfg[9], kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self,x):
        x1 = self.head(x)        ##96,512,512
        x2 = self.layer1(x1)      ##128,512,512
        x1d = self.head_l1(x1)      ## 128,512,512
        x3 = torch.add(x1d,x2)      ##128,512,512
        x4 = self.layer2(x3)        ## 512,256,256
        x2d = self.l1_l2(x2)        ##512,256,256
        x5 = torch.add(x4,x2d)      ##512,256,256
        x6 = self.layer3(x5)      ##256,128,128
        x3d = self.l2_l3(x4)       ## 256,128,128
        x7 = torch.add(x6,x3d)    ## 256,128,128
        x4d = self.head_l3(x1)     ## 96,128,128
        x8 = torch.cat((x4d,x7),dim=1) ##256+96,128,128
        x9 = self.layer4(x8)    ## 256,128,128
        x10 = self.A3D1(x9)    ##128,128,128
        x11 = self.A3D2(x10)   ##64,64,64
        x12 = self.A3D3(torch.cat((x11,F.max_pool2d(x10,2,2)),dim=1))   ## 128,64,64
        x13 = self._3t4_1(x12)  ##64,64,64,2
        x14 = self.A4D1(x13)    ##64,64,64,2
        x15 = self._4t3_1(x14)  ##32,64,64
        #x15 = torch.add(F.max_pool2d(self.wl2(x7), 2, 2), x15)
        #x16 = self.tail1(x15)   ##nc,1,3

        x17 = torch.add(x11,self.bridge1(x15)) ##64,64,64
        x18 = self._3t4_2(x17) ## 16,64,64,4
        x19 = self.A4D2(torch.add(self.up_sample1(x14.view(x18.shape[0],-1,x18.shape[2],x18.shape[3],x18.shape[4])),x18)) ##64,64,64,4
        x20 = self._4t3_2(x19) ##64,64,64
        #x20 =torch.add(F.max_pool2d(self.wl1(x5),4,4),x20)
        #x21 = self.tail2(x20)  ##nc,1,3

        x22 = torch.add(x10,self.bridge2(F.interpolate(x20, size=x10.shape[2:4], mode='bilinear',align_corners=True)))  ##128,128,128
        x23 = self._3t4_3(x22) ##16,128,128,8
        x24 = self.A4D3(torch.add(self.up_sample2(F.interpolate(x19,size=x23.shape[2:5], mode='trilinear',align_corners=True)),x23))  ##64,128,128,8
        x25 = self._4t3_3(x24)   ##128,128,128
        #x25 = torch.add(F.max_pool2d(x3, 4, 4),x25)
        x26 = self.tail3(x25)  ##nc,1,3

        if self.tail_add :
            x21 = self.tail2(torch.add(self.tailadd1(x25),x20))
            x16 = self.tail1(torch.add(self.tailadd2(x20),x15))
        else:
            x16 = self.tail1(x15)  ##nc,1,3
            x21 = self.tail2(x20)  ##nc,1,3

        if self.deepsuper:
            return [[0.2, x16], [0.4, x21]], x26
        else:
            return x26

