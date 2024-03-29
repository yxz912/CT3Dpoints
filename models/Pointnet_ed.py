import torch
from torch import nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
from configs.config_setting import setting_config

def find_positions(matrix, value):
    positions = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j].item() < value:
                positions.append((i, j))
    return positions

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

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPP, self).__init__()
        self.aspp_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.aspp_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=rates[0], dilation=rates[0]),
            nn.ReLU(inplace=True)
        )
        self.aspp_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=rates[1], dilation=rates[1]),
            nn.ReLU(inplace=True)
        )
        self.aspp_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=rates[2], dilation=rates[2]),
            nn.ReLU(inplace=True)
        )
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d(64),
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv_1x1_output = nn.Conv2d(out_channels * 5, out_channels, 1)

    def forward(self, x):
        x_1 = self.aspp_1(x)
        x_6 = self.aspp_6(x)
        x_12 = self.aspp_12(x)
        x_18 = self.aspp_18(x)

        x_image_pool = self.aspp_pool(x)
        x_image_pool = F.upsample_bilinear(x_image_pool, size=x.size()[2:])

        x_out = torch.cat([x_1, x_6, x_12, x_18, x_image_pool], dim=1)
        x_out = self.conv_1x1_output(x_out)

        return x_out

# class Pointneted(nn.Module):
#     def __init__(self,num_classes=3,input_channels=64,cfg=[32,64,96,128,256,128,256,128,64,32,16],deep_supervision=True,tailadd=True,data=True):
#         super().__init__()
#
#         self.deepsuper = deep_supervision
#         self.tail_add = tailadd
#         self.num_classes = num_classes
#         self.data = data
#
#         self.aspp11 = ASPP(in_channels=input_channels,out_channels=cfg[2],rates=[1,3,5,7])
#
#         self.aspp12 = nn.Sequential(nn.MaxPool2d(2,2),
#                                    ASPP(in_channels=input_channels, out_channels=cfg[1], rates=[1, 3, 5, 7]),
#                                    nn.ConvTranspose2d(cfg[1], cfg[1], kernel_size=3, stride=2, padding=1, output_padding=1,groups=cfg[1])
#         )
#
#         self.aspp13 = nn.Sequential(nn.MaxPool2d(4, 4),
#                                    ASPP(in_channels=input_channels, out_channels=cfg[0], rates=[1, 3, 5, 7]),
#                                    nn.ConvTranspose2d(cfg[0], cfg[0], kernel_size=4, stride=4, groups=cfg[0])
#                                    )
#
#         self.aspp21 = ASPP(in_channels=cfg[0]+cfg[1]+cfg[2], out_channels=cfg[5], rates=[1, 3, 5, 7])
#
#         self.aspp22 = nn.Sequential(nn.MaxPool2d(2, 2),
#                                    ASPP(in_channels=cfg[0]+cfg[1]+cfg[2], out_channels=cfg[4], rates=[1, 3, 5, 7]),
#                                    nn.ConvTranspose2d(cfg[4], cfg[4], kernel_size=3, stride=2, padding=1,
#                                                       output_padding=1, groups=cfg[4])
#                                    )
#
#         self.aspp23 = nn.Sequential(nn.MaxPool2d(4, 4),
#                                    ASPP(in_channels=cfg[0]+cfg[1]+cfg[2], out_channels=cfg[3], rates=[1, 3, 5, 7]),
#                                    nn.ConvTranspose2d(cfg[3], cfg[3], kernel_size=4, stride=4, groups=cfg[3])
#                                    )
#
#         self.aspd = nn.Conv2d(cfg[3]+cfg[4]+cfg[5],cfg[6],kernel_size=3,stride=1,padding=1)
#
#         self.head = nn.Sequential(nn.Conv2d(input_channels,cfg[0],kernel_size=3,stride=1,padding=1),
#                                     nn.GELU(),
#         )
#
#         self.layer1 = nn.Sequential(nn.Conv2d(cfg[0],cfg[0],kernel_size=3,stride=1,padding=1,groups=cfg[0]),
#                                     nn.GELU(),
#                                     nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1),
#                                     nn.GELU(),
#                                     nn.BatchNorm2d(cfg[1])
#         )
#
#         self.layer2 = nn.Sequential(nn.Conv2d(cfg[1],cfg[1],kernel_size=3,stride=1,padding=1,groups=cfg[1]),
#                                     nn.GELU(),
#                                     nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1),
#                                     nn.GELU(),
#                                     nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1),
#                                     nn.GELU(),
#                                     nn.BatchNorm2d(cfg[3])
#         )
#
#         self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[3], kernel_size=3, stride=1, padding=1,groups=cfg[3]),
#                                     nn.GELU(),
#                                     nn.Conv2d(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1),
#                                     nn.GELU(),
#                                     nn.Conv2d(cfg[4], cfg[5], kernel_size=3, stride=1, padding=1),
#                                     nn.GELU(),
#                                     nn.Conv2d(cfg[5], cfg[6], kernel_size=3, stride=2, padding=1),
#                                     nn.GELU(),
#                                     nn.BatchNorm2d(cfg[6])
#         )
#
#         self.layer4 = nn.Sequential(nn.Conv2d(cfg[6]+cfg[0], cfg[5], kernel_size=3, stride=1, padding=1),
#                                     nn.GELU(),
#                                     Deepwisenn(cfg[5],cfg[6]),
#         )
#
#         self.A3D1 = nn.Sequential( Self_Attention_3D(2*cfg[6], cfg[6]),
#                                    nn.Conv2d(3*cfg[6],cfg[7],kernel_size=3,stride=1,padding=1),
#                                    nn.BatchNorm2d(cfg[7])
#         )
#
#         self.A3D2 = nn.Sequential( Self_Attention_3D(cfg[7], cfg[7]),
#                                    nn.Conv2d(2*cfg[7],cfg[8],kernel_size=3,stride=2,padding=1),
#                                    nn.BatchNorm2d(cfg[8])
#         )
#
#         self.A3D3 = nn.Sequential(Self_Attention_3D(cfg[8]+cfg[7], cfg[7]),
#                                   nn.Conv2d(cfg[8]+2*cfg[7], cfg[7], kernel_size=3, stride=1, padding=1),
#                                   nn.BatchNorm2d(cfg[7])
#                                   )
#
#         self.A4D1 = nn.Sequential(Self_Attention_4D(cfg[8], cfg[9]),
#                                   nn.Conv3d(cfg[8]+cfg[9], cfg[8], kernel_size=3, stride=1, padding=1),
#                                   nn.BatchNorm3d(cfg[8])
#                                   )
#
#         self.A4D2 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
#                                   nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
#                                   nn.BatchNorm3d(cfg[8])
#                                   )
#
#         self.A4D3 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
#                                   nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
#                                   nn.BatchNorm3d(cfg[8])
#                                   )
#
#         self._3t4_1 = Ascension_3t4D(cfg[7],cfg[8])
#
#         self._3t4_2 = Ascension_3t4D(cfg[8],cfg[10])
#
#         self._3t4_3 = Ascension_3t4D(cfg[7],cfg[10])
#
#         self._4t3_1 = lower_4t3D(cfg[8],cfg[9],2)
#
#         self._4t3_2 = lower_4t3D(cfg[8],cfg[8],4)
#
#         self._4t3_3 = lower_4t3D(cfg[8],cfg[7],8)
#
#         self.head_l1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1)
#
#         self.l1_l2 = nn.Conv2d(cfg[1], cfg[3], kernel_size=3,stride=2,padding=1)
#
#         self.l2_l3 = nn.Conv2d(cfg[3], cfg[6], kernel_size=3,stride=2,padding=1)
#
#         self.wl1 = nn.Conv2d(cfg[3],cfg[8],kernel_size=1,stride=1)
#
#         self.wl2 = nn.Conv2d(cfg[6],cfg[9],kernel_size=1,stride=1)
#
#         self.head_l3 = nn.MaxPool2d(kernel_size=4)
#
#         self.bridge1 = nn.Conv2d(cfg[9],cfg[8],kernel_size=1)
#
#         self.bridge2 = nn.Conv2d(cfg[8],cfg[7],kernel_size=1)
#
#         self.up_sample1 = nn.Conv3d(cfg[9],cfg[10],kernel_size=1)
#
#         self.up_sample2 = nn.Conv3d(cfg[8],cfg[10],kernel_size=1)
#
#         self.tail1 = nn.Sequential(nn.GroupNorm(4, cfg[9]),
#                                    nn.Conv2d(cfg[9], num_classes, kernel_size=3, stride=2, padding=1),
#                                    nn.AdaptiveAvgPool2d((1,3))
#         )
#
#         self.tail2 = nn.Sequential(nn.GroupNorm(4, cfg[8]),
#                                    nn.Conv2d(cfg[8], cfg[10], kernel_size=3, stride=2, padding=1),
#                                    nn.Conv2d(cfg[10], num_classes, kernel_size=3, stride=2, padding=1),
#                                    nn.AdaptiveAvgPool2d((1,3))
#         )
#
#         self.tail3 = nn.Sequential(nn.GroupNorm(4, cfg[7]),
#                                    nn.Conv2d(cfg[7], cfg[8], kernel_size=3, stride=2, padding=1),
#                                    nn.GELU(),
#                                    nn.Conv2d(cfg[8], cfg[10], kernel_size=3, stride=2, padding=1),
#                                    nn.Conv2d(cfg[10], num_classes, kernel_size=3, stride=2, padding=1),
#                                    nn.AdaptiveAvgPool2d((1,3))
#         )
#
#         self.tailadd1 = nn.Conv2d(cfg[7], cfg[8], kernel_size=3, stride=2, padding=1)
#
#         self.tailadd2 = nn.Conv2d(cfg[8], cfg[9], kernel_size=1)
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Conv1d):
#             n = m.kernel_size[0] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#         elif isinstance(m, nn.Conv3d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self,x):
#         xp11 = self.aspp11(x)
#         xp12 = self.aspp12(x)
#         xp13 = self.aspp13(x)
#         xp1 = torch.cat([xp11,xp12,xp13],dim=1)
#
#         xp21 = self.aspp21(xp1)
#         xp22 = self.aspp22(xp1)
#         xp23 = self.aspp23(xp1)
#         xp2 = torch.cat([xp21, xp22, xp23], dim=1)
#
#         x1 = self.head(x)        ##96,512,512
#         x2 = self.layer1(x1)      ##128,512,512
#         x1d = self.head_l1(x1)      ## 128,512,512
#         x3 = torch.add(x1d,x2)      ##128,512,512
#         x4 = self.layer2(x3)        ## 512,256,256
#         x2d = self.l1_l2(x2)        ##512,256,256
#         x5 = torch.add(x4,x2d)      ##512,256,256
#         x6 = self.layer3(x5)      ##256,128,128
#         x3d = self.l2_l3(x4)       ## 256,128,128
#         x7 = torch.add(x6,x3d)    ## 256,128,128
#         x4d = self.head_l3(x1)     ## 96,128,128
#         x8 = torch.cat((x4d,x7),dim=1) ##256+96,128,128
#         x9 = self.layer4(x8)    ## 256,128,128
#
#         xp = F.interpolate(self.aspd(xp2),size=x9.shape[2:4],mode='bilinear',align_corners=True)
#         xp9 = torch.cat([x9,xp],dim=1)
#
#         x10 = self.A3D1(xp9)    ##128,128,128
#         x11 = self.A3D2(x10)   ##64,64,64
#         x12 = self.A3D3(torch.cat((x11,F.max_pool2d(x10,2,2)),dim=1))   ## 128,64,64
#         x13 = self._3t4_1(x12)  ##64,64,64,2
#         x14 = self.A4D1(x13)    ##64,64,64,2
#         x15 = self._4t3_1(x14)  ##32,64,64
#         #x15 = torch.add(F.max_pool2d(self.wl2(x7), 2, 2), x15)
#         #x16 = self.tail1(x15)   ##nc,1,3
#
#         x17 = torch.add(x11,self.bridge1(x15)) ##64,64,64
#         x18 = self._3t4_2(x17) ## 16,64,64,4
#         x19 = self.A4D2(torch.add(self.up_sample1(x14.view(x18.shape[0],-1,x18.shape[2],x18.shape[3],x18.shape[4])),x18)) ##64,64,64,4
#         x20 = self._4t3_2(x19) ##64,64,64
#         #x20 =torch.add(F.max_pool2d(self.wl1(x5),4,4),x20)
#         #x21 = self.tail2(x20)  ##nc,1,3
#
#         x22 = torch.add(x10,self.bridge2(F.interpolate(x20, size=x10.shape[2:4], mode='bilinear',align_corners=True)))  ##128,128,128
#         x23 = self._3t4_3(x22) ##16,128,128,8
#         x24 = self.A4D3(torch.add(self.up_sample2(F.interpolate(x19,size=x23.shape[2:5], mode='trilinear',align_corners=True)),x23))  ##64,128,128,8
#         x25 = self._4t3_3(x24)   ##128,128,128
#         #x25 = torch.add(F.max_pool2d(x3, 4, 4),x25)
#         x26 = self.tail3(x25)  ##nc,1,3
#
#         if self.tail_add :
#             x21 = self.tail2(torch.add(self.tailadd1(x25),x20))
#             x16 = self.tail1(torch.add(self.tailadd2(x20),x15))
#             all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + \
#                           list(self.tail1.parameters()) + list(self.tailadd2.parameters())+ list(self.A3D3.parameters())
#             all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
#                           list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) \
#                           + list(self.tail2.parameters()) + list(self.tailadd1.parameters())+list(self.A3D2.parameters())
#         else:
#             x16 = self.tail1(x15)  ##nc,1,3
#             x21 = self.tail2(x20)  ##nc,1,3
#             all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + list(self.tail1.parameters())+ list(self.A3D3.parameters())
#             all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
#                           list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) + list(self.tail2.parameters())+list(self.A3D2.parameters())
#
#         all_params1 = list(self.head.parameters()) + list(self.layer1.parameters()) + list(self.head_l1.parameters()) +\
#                       list(self.layer2.parameters())+list(self.l1_l2.parameters()) + list(self.layer3.parameters())+\
#                       list(self.l2_l3.parameters()) + list(self.head_l3.parameters())+ list(self.layer4.parameters())+\
#                       list(self.aspp11.parameters())+list(self.aspp12.parameters())+list(self.aspp13.parameters())+ \
#                       list(self.aspp21.parameters())+list(self.aspp22.parameters())+list(self.aspp23.parameters())+list(self.aspd.parameters())
#
#
#         all_params5 = list(self.bridge2.parameters()) + list(self._3t4_3.parameters())+list(self.A4D3.parameters()) + list(self.up_sample2.parameters()) \
#                       + list(self._4t3_3.parameters())+list(self.tail3.parameters())+list(self.A3D1.parameters())
#
#         # 计算L2正则化项
#         l2_reg=[]
#         l2_loss=0.0
#         l2_reg.append(sum(param.norm(2)**2 for param in all_params1)**0.5)
#         l2_reg.append(sum(param.norm(2)**2 for param in all_params3)**0.5)
#         l2_reg.append(sum(param.norm(2)**2 for param in all_params4)**0.5)
#         l2_reg.append(sum(param.norm(2)**2 for param in all_params5)**0.5)
#
#         for i in l2_reg:
#             l2_loss += (i/sum(l2_reg)) * i
#             #print(i/sum(l2_reg)*i)
#             #l2_loss += i
#         #print("|***--L2 LOSS====>:",l2_loss)
#         if self.data:
#             point_mean26 = torch.mean(x26, dim=3)
#             point_mean16 = torch.mean(x16, dim=3)
#             point_mean21 = torch.mean(x21, dim=3)
#
#             result26 = find_positions(point_mean26, 1)
#             result16 = find_positions(point_mean16, 1)
#             result21 = find_positions(point_mean21, 1)
#
#             for t in result26:
#                 x26[t[0]][t[1]] = torch.tensor([-1., -1., -1.])
#             for t in result21:
#                 x21[t[0]][t[1]] = torch.tensor([-1., -1., -1.])
#             for t in result16:
#                 x16[t[0]][t[1]] = torch.tensor([-1., -1., -1.])
#
#         if self.deepsuper:
#             return [l2_loss,[0.1, x16], [0.2, x21]], x26
#         else:
#             return x26

class Pointneted_plus(nn.Module):
    def __init__(self,num_classes=3,input_channels=64,cfg=[96,128,256,512,1024,512,256,128,64,32,16],deep_supervision=True,tailadd=True):
        super().__init__()

        self.deepsuper = deep_supervision
        self.tail_add = tailadd
        self.num_classes = num_classes
        #self.weight = nn.Parameter(torch.ones(4))
        #self.l2w = nn.Parameter(torch.ones(4))

        #self.aspp = ASPP(in_channels=input_channels,out_channels=input_channels,rates=[1,3,5,7])

        self.head = nn.Sequential(nn.Conv2d(input_channels,cfg[0],kernel_size=3,stride=1,padding=1),
                                    nn.GELU(),
        )

        self.layer1 = nn.Sequential(nn.Conv2d(cfg[0],cfg[0],kernel_size=3,stride=1,padding=1,groups=cfg[0]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[1], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[1])
        )

        self.layer2 = nn.Sequential(nn.Conv2d(cfg[1],cfg[1],kernel_size=3,stride=1,padding=1,groups=cfg[1]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[3], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[3])
        )

        self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[3], kernel_size=3, stride=1, padding=1,groups=cfg[3]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[4], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[5], cfg[6], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[6], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[6])
        )

        self.layer4 = nn.Sequential(nn.Conv2d(cfg[6]+cfg[0], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    Deepwisenn(cfg[5],cfg[6]),
        )

        self.A3D1 = nn.Sequential( Self_Attention_3D(cfg[6], cfg[6]),
                                   nn.Conv2d(2*cfg[6],cfg[7],kernel_size=3,stride=1,padding=1),
                                   #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[7])
        )

        self.A3D2 = nn.Sequential( Self_Attention_3D(cfg[7], cfg[7]),
                                   nn.Conv2d(2*cfg[7],cfg[8],kernel_size=3,stride=2,padding=1),
                                   #LayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[8])
        )

        self.A3D3 = nn.Sequential(Self_Attention_3D(cfg[8]+cfg[7], cfg[7]),
                                  nn.Conv2d(cfg[8]+2*cfg[7], cfg[7], kernel_size=3, stride=1, padding=1),
                                  #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm2d(cfg[7])
                                  )

        self.A4D1 = nn.Sequential(Self_Attention_4D(cfg[8], cfg[9]),
                                  nn.Conv3d(cfg[8]+cfg[9], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D2 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D3 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
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
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + \
                          list(self.tail1.parameters()) + list(self.tailadd2.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) \
                          + list(self.tail2.parameters()) + list(self.tailadd1.parameters())+list(self.A3D2.parameters())
        else:
            x16 = self.tail1(x15)  ##nc,1,3
            x21 = self.tail2(x20)  ##nc,1,3
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + list(self.tail1.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) + list(self.tail2.parameters())+list(self.A3D2.parameters())

        all_params1 = list(self.head.parameters()) + list(self.layer1.parameters()) + list(self.head_l1.parameters()) +\
                      list(self.layer2.parameters())+list(self.l1_l2.parameters()) + list(self.layer3.parameters())+\
                      list(self.l2_l3.parameters()) + list(self.head_l3.parameters())+ list(self.layer4.parameters())+\
                      list(self.A3D1.parameters())+list(self.A3D2.parameters())+list(self.A3D3.parameters())


        all_params5 = list(self.bridge2.parameters()) + list(self._3t4_3.parameters())+list(self.A4D3.parameters()) + list(self.up_sample2.parameters()) \
                      + list(self._4t3_3.parameters())+list(self.tail3.parameters())+list(self.A3D1.parameters())

        # 计算L2正则化项
        l2_reg=[]
        l2_loss=0.0
        l2_reg.append(sum(param.norm(2)**2 for param in all_params1)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params3)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params4)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params5)**0.5)

        for i in l2_reg:
            l2_loss += (i/sum(l2_reg)) * i
            #print(i/sum(l2_reg)*i)
            #l2_loss += i
        #print("|***--L2 LOSS====>:",l2_loss)

        if self.deepsuper:
            return [l2_loss,[0.1, x16], [0.2, x21]], x26
        else:
            return x26

class Pointneted_plus_mmld(nn.Module):
    def __init__(self,num_classes=3,input_channels=64,cfg=[96,128,256,512,1024,512,256,128,64,32,16],deep_supervision=True,tailadd=True):
        super().__init__()

        self.deepsuper = deep_supervision
        self.tail_add = tailadd
        self.num_classes = num_classes
        #self.weight = nn.Parameter(torch.ones(4))
        #self.l2w = nn.Parameter(torch.ones(4))

        #self.aspp = ASPP(in_channels=input_channels,out_channels=input_channels,rates=[1,3,5,7])

        self.head = nn.Sequential(nn.Conv2d(input_channels,cfg[0],kernel_size=3,stride=1,padding=1),
                                    nn.GELU(),
        )

        self.layer1 = nn.Sequential(nn.Conv2d(cfg[0],cfg[0],kernel_size=3,stride=1,padding=1,groups=cfg[0]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[1], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[1])
        )

        self.layer2 = nn.Sequential(nn.Conv2d(cfg[1],cfg[1],kernel_size=3,stride=1,padding=1,groups=cfg[1]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[3], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[3])
        )

        self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[3], kernel_size=3, stride=1, padding=1,groups=cfg[3]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[4], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[5], cfg[6], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[6], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[6])
        )

        self.layer4 = nn.Sequential(nn.Conv2d(cfg[6]+cfg[0], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    Deepwisenn(cfg[5],cfg[6]),
        )

        self.A3D1 = nn.Sequential( Self_Attention_3D(cfg[6], cfg[6]),
                                   nn.Conv2d(2*cfg[6],cfg[7],kernel_size=3,stride=1,padding=1),
                                   #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[7])
        )

        self.A3D2 = nn.Sequential( Self_Attention_3D(cfg[7], cfg[7]),
                                   nn.Conv2d(2*cfg[7],cfg[8],kernel_size=3,stride=2,padding=1),
                                   #LayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[8])
        )

        self.A3D3 = nn.Sequential(Self_Attention_3D(cfg[8]+cfg[7], cfg[7]),
                                  nn.Conv2d(cfg[8]+2*cfg[7], cfg[7], kernel_size=3, stride=1, padding=1),
                                  #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm2d(cfg[7])
                                  )

        self.A4D1 = nn.Sequential(Self_Attention_4D(cfg[8], cfg[9]),
                                  nn.Conv3d(cfg[8]+cfg[9], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D2 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D3 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
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
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + \
                          list(self.tail1.parameters()) + list(self.tailadd2.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) \
                          + list(self.tail2.parameters()) + list(self.tailadd1.parameters())+list(self.A3D2.parameters())
        else:
            x16 = self.tail1(x15)  ##nc,1,3
            x21 = self.tail2(x20)  ##nc,1,3
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + list(self.tail1.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) + list(self.tail2.parameters())+list(self.A3D2.parameters())

        all_params1 = list(self.head.parameters()) + list(self.layer1.parameters()) + list(self.head_l1.parameters()) +\
                      list(self.layer2.parameters())+list(self.l1_l2.parameters()) + list(self.layer3.parameters())+\
                      list(self.l2_l3.parameters()) + list(self.head_l3.parameters())+ list(self.layer4.parameters())+\
                      list(self.A3D1.parameters())+list(self.A3D2.parameters())+list(self.A3D3.parameters())


        all_params5 = list(self.bridge2.parameters()) + list(self._3t4_3.parameters())+list(self.A4D3.parameters()) + list(self.up_sample2.parameters()) \
                      + list(self._4t3_3.parameters())+list(self.tail3.parameters())+list(self.A3D1.parameters())

        # 计算L2正则化项
        l2_reg=[]
        l2_loss=0.0
        l2_reg.append(sum(param.norm(2)**2 for param in all_params1)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params3)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params4)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params5)**0.5)

        for i in l2_reg:
            l2_loss += (i/sum(l2_reg)) * i
            #print(i/sum(l2_reg)*i)
            #l2_loss += i
        #print("|***--L2 LOSS====>:",l2_loss)

        point_mean26 = torch.mean(x26, dim=3)
        point_mean16 = torch.mean(x16, dim=3)
        point_mean21 = torch.mean(x21, dim=3)

        result26 = find_positions(point_mean26, 2)
        result16 = find_positions(point_mean16, 2)
        result21 = find_positions(point_mean21, 2)

        for t in result26:
            x26[t[0]][t[1]] = torch.tensor([-1., -1., -1.])
        for t in result21:
            x21[t[0]][t[1]] = torch.tensor([-1., -1., -1.])
        for t in result16:
            x16[t[0]][t[1]] = torch.tensor([-1., -1., -1.])

        if self.deepsuper:
            return [l2_loss,[0.1, x16], [0.2, x21]], x26
        else:
            return x26

class Pointnet_ed(nn.Module):
    def __init__(self,num_classes=3,input_channels=64,cfg=[96,128,256,512,1024,512,256,128,64,32,16],deep_supervision=True,tailadd=True):
        super().__init__()

        self.deepsuper = deep_supervision
        self.tail_add = tailadd
        self.num_classes = num_classes

        self.head = nn.Sequential(nn.Conv2d(input_channels,cfg[0],kernel_size=3,stride=1,padding=1),
                                    nn.GELU(),
        )

        self.layer1 = nn.Sequential(nn.Conv2d(cfg[0],cfg[0],kernel_size=3,stride=1,padding=1,groups=cfg[0]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[1], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[1])
        )

        self.layer2 = nn.Sequential(nn.Conv2d(cfg[1],cfg[1],kernel_size=3,stride=1,padding=1,groups=cfg[1]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[3], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[3])
        )

        self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[3], kernel_size=3, stride=1, padding=1,groups=cfg[3]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[4], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[5], cfg[6], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[6], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[6])
        )

        self.layer4 = nn.Sequential(nn.Conv2d(cfg[6]+cfg[0], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    Deepwisenn(cfg[5],cfg[6]),
        )

        self.A3D1 = nn.Sequential( Self_Attention_3D(cfg[6], cfg[6]),
                                   nn.Conv2d(2*cfg[6],cfg[7],kernel_size=3,stride=1,padding=1),
                                   #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[7])
        )

        self.A3D2 = nn.Sequential( Self_Attention_3D(cfg[7], cfg[7]),
                                   nn.Conv2d(2*cfg[7],cfg[8],kernel_size=3,stride=2,padding=1),
                                   #LayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[8])
        )

        self.A3D3 = nn.Sequential(Self_Attention_3D(cfg[8]+cfg[7], cfg[7]),
                                  nn.Conv2d(cfg[8]+2*cfg[7], cfg[7], kernel_size=3, stride=1, padding=1),
                                  #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm2d(cfg[7])
                                  )

        self.A4D1 = nn.Sequential(Self_Attention_4D(cfg[8], cfg[9]),
                                  nn.Conv3d(cfg[8]+cfg[9], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D2 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D3 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
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
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + \
                          list(self.tail1.parameters()) + list(self.tailadd2.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) \
                          + list(self.tail2.parameters()) + list(self.tailadd1.parameters())+list(self.A3D2.parameters())
        else:
            x16 = self.tail1(x15)  ##nc,1,3
            x21 = self.tail2(x20)  ##nc,1,3
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + list(self.tail1.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) + list(self.tail2.parameters())+list(self.A3D2.parameters())

        all_params1 = list(self.head.parameters()) + list(self.layer1.parameters()) + list(self.head_l1.parameters()) +\
                      list(self.layer2.parameters())+list(self.l1_l2.parameters()) + list(self.layer3.parameters())+\
                      list(self.l2_l3.parameters()) + list(self.head_l3.parameters())+ list(self.layer4.parameters())+\
                      list(self.A3D1.parameters())+list(self.A3D2.parameters())+list(self.A3D3.parameters())

        all_params5 = list(self.bridge2.parameters()) + list(self._3t4_3.parameters())+list(self.A4D3.parameters()) + list(self.up_sample2.parameters()) \
                      + list(self._4t3_3.parameters())+list(self.tail3.parameters())+list(self.A3D1.parameters())

        # 计算L2正则化项
        l2_reg=[]
        l2_loss=0.0
        l2_reg.append(sum(param.norm(2)**2 for param in all_params1)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params3)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params4)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params5)**0.5)

        for i in l2_reg:
            l2_loss += (i/sum(l2_reg)) * i

        xyz1 = torch.squeeze(x16, dim=2)
        x16 = torch.unsqueeze(self.Relevance_adjustments(xyz1),dim=2)
        xyz2 = torch.squeeze(x21, dim=2)
        x21 = torch.unsqueeze(self.Relevance_adjustments(xyz2),dim=2)
        xyz = torch.squeeze(x26,dim=2)
        x26 = torch.unsqueeze(self.Relevance_adjustments(xyz),dim=2)

        if self.deepsuper:
            return [l2_loss,[0.1, x16], [0.2, x21]], x26
        else:
            return x26

    def Relevance_adjustments(self,xyz):
        # a1_opt = torch.tensor([[-0.01171973, 0.00511874, -0.03102786]])
        # a2_opt = torch.tensor([[-0.01537341, 0.0074207, 0.00215751]])
        # a3_opt = torch.tensor([[-0.00891788, 0.00088328, 0.02658808]])
        # a4_opt = torch.tensor([[0.01766429, -0.00893865, -0.00241741]])
        # a5_opt = torch.tensor([[0.0211294, 0.00557574, 0.00478809]])
        # a6_opt = torch.tensor([[-0.00246022, -0.0091335, -0.00135202]])
        # b = torch.tensor(-0.18116703).cuda()

        a1_opt=torch.tensor([[-0.00224888, - 0.0024509 ,  0.0031326]])
        a2_opt=torch.tensor([[0.01155325 , 0.00028369 ,- 0.00083645]])
        a3_opt=torch.tensor([[-0.00208301 , 0.00097889, - 0.00154358]])
        a4_opt=torch.tensor([[-0.00660129 , 0.00106194, - 0.0001622]])
        a5_opt=torch.tensor([[-0.00130171 , 0.00116966, - 0.00068797]])
        a6_opt=torch.tensor([[0.00042264, - 0.00102405 , 0.00025911]])
        b = torch.tensor(0.06752735).cuda()

        mean1 = [85.38860869565218, 42.480782608695655 ,100.6086956521739]
        std1 = [19.910707448700823, 9.613875221721361, 8.090462073302508]
        mean2 = [59.140608695652176, 41.85843478260869 ,101.49565217391304]
        std2 = [19.193310124690143, 10.816812645644006 ,8.226546544749507]
        mean3 = [59.077999999999996 ,42.2328695652174, 101.6]
        std3 = [19.56582149521478, 10.853970843426072 ,8.433679892060116]
        mean4 = [21.51991304347826, 58.56895652173912 ,99.77391304347826]
        std4 = [17.483606109757055, 15.138758054506724, 8.966663738600563]
        mean5 = [140.95017391304347 ,37.60078260869565 ,101.03478260869565]
        std5 = [21.507908918737563, 10.001301232541962, 8.219732815696423]
        mean6 = [137.40017391304346, 101.50678260869566 ,99.73913043478261]
        std6 = [19.25242024232656 ,10.279309530848941 ,3.934434286768532]

        mean=torch.tensor([mean1,mean2,mean3,mean4,mean5,mean6]).cuda()
        std=torch.tensor([std1,std2,std3,std4,std5,std6]).cuda()
        a = torch.unsqueeze(torch.cat((a1_opt, a2_opt, a3_opt, a4_opt, a5_opt, a6_opt),dim=0),dim=0).cuda()

        relev_loss = torch.zeros(xyz.shape[0]).cuda()
        new_xyz = torch.zeros(xyz.shape)
        for kd,data in enumerate(xyz):
            y = torch.sum(torch.sum(a * data, axis=2))-b
            last_y=100
            last_data = data.cuda()
            for i in range(10):
                for j,jp in enumerate(a[0]):
                    dot = y * jp.cuda()*(abs(sum((data[j]-mean[j])*std[j])))
                    new_data = data[j] - dot
                    new_xyz[kd,j] = new_data
                new_y = torch.sum(torch.sum(a * new_xyz[kd].cuda(), axis=2)) - b
                #print("%d,lossdd===%f"%(i,new_y))
                if abs(new_y)>=abs(last_y):
                    new_xyz[kd] = last_data
                    break
                else:
                    data = new_xyz[kd].cuda()
                    last_data = new_xyz[kd].cuda()
                    last_y = new_y
                    y = new_y

            relev_loss[kd] = new_y
        # dis = torch.unsqueeze(relev_loss,dim=1)
        #print("lastdata=",xyz,"newdata=",new_xyz)
        return new_xyz.cuda()

class Pointneted(nn.Module):
    def __init__(self,num_classes=3,input_channels=64,cfg=[96,128,256,512,1024,512,256,128,64,32,16],deep_supervision=True,tailadd=True):
        super().__init__()

        self.deepsuper = deep_supervision
        self.tail_add = tailadd
        self.num_classes = num_classes

        self.head = nn.Sequential(nn.Conv2d(input_channels,cfg[0],kernel_size=3,stride=1,padding=1),
                                    nn.GELU(),
        )

        self.layer1 = nn.Sequential(nn.Conv2d(cfg[0],cfg[0],kernel_size=3,stride=1,padding=1,groups=cfg[0]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[1], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[1])
        )

        self.layer2 = nn.Sequential(nn.Conv2d(cfg[1],cfg[1],kernel_size=3,stride=1,padding=1,groups=cfg[1]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[3], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[3])
        )

        self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[3], kernel_size=3, stride=1, padding=1,groups=cfg[3]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[4], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[5], cfg[6], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    #LayerNorm(cfg[6], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[6])
        )

        self.layer4 = nn.Sequential(nn.Conv2d(cfg[6]+cfg[0], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    Deepwisenn(cfg[5],cfg[6]),
        )

        self.A3D1 = nn.Sequential( Self_Attention_3D(cfg[6], cfg[6]),
                                   nn.Conv2d(2*cfg[6],cfg[7],kernel_size=3,stride=1,padding=1),
                                   #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[7])
        )

        self.A3D2 = nn.Sequential( Self_Attention_3D(cfg[7], cfg[7]),
                                   nn.Conv2d(2*cfg[7],cfg[8],kernel_size=3,stride=2,padding=1),
                                   #LayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                   nn.BatchNorm2d(cfg[8])
        )

        self.A3D3 = nn.Sequential(Self_Attention_3D(cfg[8]+cfg[7], cfg[7]),
                                  nn.Conv2d(cfg[8]+2*cfg[7], cfg[7], kernel_size=3, stride=1, padding=1),
                                  #LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm2d(cfg[7])
                                  )

        self.A4D1 = nn.Sequential(Self_Attention_4D(cfg[8], cfg[9]),
                                  nn.Conv3d(cfg[8]+cfg[9], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D2 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D3 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  #Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
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

        self.top_1 = nn.Conv3d(cfg[10],cfg[8],1)

        self.top_2 = nn.Conv3d(cfg[8],cfg[8],1)

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

        x17 = torch.add(x11,self.bridge1(x15)) ##64,64,64
        x18 = self._3t4_2(x17) ## 16,64,64,4
        x19 = self.A4D2(torch.add(self.up_sample1(x14.view(x18.shape[0],-1,x18.shape[2],x18.shape[3],x18.shape[4])),x18)) ##64,64,64,4

        x19 = torch.add(F.interpolate(self.top_2(x13),size=x19.shape[2:5], mode='trilinear',align_corners=True),x19)

        x20 = self._4t3_2(x19) ##64,64,64

        x22 = torch.add(x10,self.bridge2(F.interpolate(x20, size=x10.shape[2:4], mode='bilinear',align_corners=True)))  ##128,128,128
        x23 = self._3t4_3(x22) ##16,128,128,8
        x24 = self.A4D3(torch.add(self.up_sample2(F.interpolate(x19,size=x23.shape[2:5], mode='trilinear',align_corners=True)),x23))  ##64,128,128,8

        x24 = torch.add(F.interpolate(self.top_1(x18),size=x24.shape[2:5], mode='trilinear',align_corners=True),x24)

        x25 = self._4t3_3(x24)   ##128,128,128
        #x25 = torch.add(F.max_pool2d(x3, 4, 4),x25)
        x26 = self.tail3(x25)  ##nc,1,3

        if self.tail_add :
            x21 = self.tail2(torch.add(self.tailadd1(x25),x20))
            x16 = self.tail1(torch.add(self.tailadd2(x20),x15))
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + \
                          list(self.tail1.parameters()) + list(self.tailadd2.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) \
                          + list(self.tail2.parameters()) + list(self.tailadd1.parameters())+list(self.A3D2.parameters())
        else:
            x16 = self.tail1(x15)  ##nc,1,3
            x21 = self.tail2(x20)  ##nc,1,3
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(self._4t3_1.parameters()) + list(self.tail1.parameters())+ list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(self._4t3_2.parameters()) + list(self.tail2.parameters())+list(self.A3D2.parameters())

        all_params1 = list(self.head.parameters()) + list(self.layer1.parameters()) + list(self.head_l1.parameters()) +\
                      list(self.layer2.parameters())+list(self.l1_l2.parameters()) + list(self.layer3.parameters())+\
                      list(self.l2_l3.parameters()) + list(self.head_l3.parameters())+ list(self.layer4.parameters())+\
                      list(self.A3D1.parameters())+list(self.A3D2.parameters())+list(self.A3D3.parameters())


        all_params5 = list(self.bridge2.parameters()) + list(self._3t4_3.parameters())+list(self.A4D3.parameters()) + list(self.up_sample2.parameters()) \
                      + list(self._4t3_3.parameters())+list(self.tail3.parameters())+list(self.A3D1.parameters())

        # 计算L2正则化项
        l2_reg=[]
        l2_loss=0.0
        l2_reg.append(sum(param.norm(2)**2 for param in all_params1)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params3)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params4)**0.5)
        l2_reg.append(sum(param.norm(2)**2 for param in all_params5)**0.5)

        for i in l2_reg:
            l2_loss += (i/sum(l2_reg)) * i

        if self.deepsuper:
            return [l2_loss,[0.1, x16], [0.2, x21]], x26
        else:
            return x26

class Pointneted_gaus(nn.Module):
    def __init__(self, num_classes=3, input_channels=64, cfg=[96, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16],
                 deep_supervision=True, tailadd=True):
        super().__init__()

        self.deepsuper = deep_supervision
        self.tail_add = tailadd
        self.num_classes = num_classes

        self.head = nn.Sequential(nn.Conv2d(input_channels, cfg[0], kernel_size=3, stride=1, padding=1),
                                  nn.GELU(),
                                  )

        self.layer1 = nn.Sequential(nn.Conv2d(cfg[0], cfg[0], kernel_size=3, stride=1, padding=1, groups=cfg[0]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    # LayerNorm(cfg[1], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[1])
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(cfg[1], cfg[1], kernel_size=3, stride=1, padding=1, groups=cfg[1]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    # LayerNorm(cfg[3], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[3])
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(cfg[3], cfg[3], kernel_size=3, stride=1, padding=1, groups=cfg[3]),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[3], cfg[4], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[4], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    nn.Conv2d(cfg[5], cfg[6], kernel_size=3, stride=2, padding=1),
                                    nn.GELU(),
                                    # LayerNorm(cfg[6], eps=1e-6, data_format='channels_first'),
                                    nn.BatchNorm2d(cfg[6])
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(cfg[6] + cfg[0], cfg[5], kernel_size=3, stride=1, padding=1),
                                    nn.GELU(),
                                    Deepwisenn(cfg[5], cfg[6]),
                                    )

        self.A3D1 = nn.Sequential(Self_Attention_3D(cfg[6], cfg[6]),
                                  nn.Conv2d(2 * cfg[6], cfg[7], kernel_size=3, stride=1, padding=1),
                                  # LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm2d(cfg[7])
                                  )

        self.A3D2 = nn.Sequential(Self_Attention_3D(cfg[7], cfg[7]),
                                  nn.Conv2d(2 * cfg[7], cfg[8], kernel_size=3, stride=2, padding=1),
                                  # LayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm2d(cfg[8])
                                  )

        self.A3D3 = nn.Sequential(Self_Attention_3D(cfg[8] + cfg[7], cfg[7]),
                                  nn.Conv2d(cfg[8] + 2 * cfg[7], cfg[7], kernel_size=3, stride=1, padding=1),
                                  # LayerNorm(cfg[7], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm2d(cfg[7])
                                  )

        self.A4D1 = nn.Sequential(Self_Attention_4D(cfg[8], cfg[9]),
                                  nn.Conv3d(cfg[8] + cfg[9], cfg[8], kernel_size=3, stride=1, padding=1),
                                  # Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D2 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  # Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self.A4D3 = nn.Sequential(Self_Attention_4D(cfg[10], cfg[10]),
                                  nn.Conv3d(2 * cfg[10], cfg[8], kernel_size=3, stride=1, padding=1),
                                  # Conv4DLayerNorm(cfg[8], eps=1e-6, data_format='channels_first')
                                  nn.BatchNorm3d(cfg[8])
                                  )

        self._3t4_1 = Ascension_3t4D(cfg[7], cfg[8])

        self._3t4_2 = Ascension_3t4D(cfg[8], cfg[10])

        self._3t4_3 = Ascension_3t4D(cfg[7], cfg[10])

        self._4t3_1 = lower_4t3D(cfg[8], cfg[9], 2)

        self._4t3_2 = lower_4t3D(cfg[8], cfg[8], 4)

        self._4t3_3 = lower_4t3D(cfg[8], cfg[7], 8)

        self.head_l1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1)

        self.l1_l2 = nn.Conv2d(cfg[1], cfg[3], kernel_size=3, stride=2, padding=1)

        self.l2_l3 = nn.Conv2d(cfg[3], cfg[6], kernel_size=3, stride=2, padding=1)

        self.wl1 = nn.Conv2d(cfg[3], cfg[8], kernel_size=1, stride=1)

        self.wl2 = nn.Conv2d(cfg[6], cfg[9], kernel_size=1, stride=1)

        self.head_l3 = nn.MaxPool2d(kernel_size=4)

        self.bridge1 = nn.Conv2d(cfg[9], cfg[8], kernel_size=1)

        self.bridge2 = nn.Conv2d(cfg[8], cfg[7], kernel_size=1)

        self.up_sample1 = nn.Conv3d(cfg[9], cfg[10], kernel_size=1)

        self.up_sample2 = nn.Conv3d(cfg[8], cfg[10], kernel_size=1)

        self.tail1 = nn.Sequential(nn.GroupNorm(4, cfg[9]),
                                   nn.Conv2d(cfg[9], cfg[10], kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(cfg[10], num_classes, kernel_size=3, stride=2, padding=1),
                                   #nn.AdaptiveAvgPool2d((1, 3))
                                   )

        self.tail2 = nn.Sequential(nn.GroupNorm(4, cfg[8]),
                                   nn.Conv2d(cfg[8], cfg[9], kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(cfg[9], cfg[10], kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(cfg[10], num_classes, kernel_size=3, stride=2, padding=1),
                                   #nn.AdaptiveAvgPool2d((1, 3))
                                   )

        self.tail3 = nn.Sequential(nn.GroupNorm(4, cfg[7]),
                                   nn.Conv2d(cfg[7], cfg[8], kernel_size=3, stride=2, padding=1),
                                   nn.GELU(),
                                   nn.Conv2d(cfg[8], cfg[9], kernel_size=3, stride=2, padding=1),
                                   nn.Conv2d(cfg[9], cfg[10], kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(cfg[10], num_classes, kernel_size=3, stride=2, padding=1),
                                   #nn.AdaptiveAvgPool2d((1, 3))
                                   )

        self.tailadd1 = nn.Conv2d(cfg[7], cfg[8], kernel_size=3, stride=2, padding=1)

        self.tailadd2 = nn.Conv2d(cfg[8], cfg[9], kernel_size=1)

        self.top_1 = nn.Conv3d(cfg[10], cfg[8], 1)

        self.top_2 = nn.Conv3d(cfg[8], cfg[8], 1)

        self.adv = nn.AdaptiveAvgPool2d((1, 3))

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

    def forward(self, x):
        x1 = self.head(x)  ##96,512,512
        x2 = self.layer1(x1)  ##128,512,512
        x1d = self.head_l1(x1)  ## 128,512,512
        x3 = torch.add(x1d, x2)  ##128,512,512
        x4 = self.layer2(x3)  ## 512,256,256
        x2d = self.l1_l2(x2)  ##512,256,256
        x5 = torch.add(x4, x2d)  ##512,256,256
        x6 = self.layer3(x5)  ##256,128,128
        x3d = self.l2_l3(x4)  ## 256,128,128
        x7 = torch.add(x6, x3d)  ## 256,128,128
        x4d = self.head_l3(x1)  ## 96,128,128
        x8 = torch.cat((x4d, x7), dim=1)  ##256+96,128,128
        x9 = self.layer4(x8)  ## 256,128,128
        x10 = self.A3D1(x9)  ##128,128,128
        x11 = self.A3D2(x10)  ##64,64,64
        x12 = self.A3D3(torch.cat((x11, F.max_pool2d(x10, 2, 2)), dim=1))  ## 128,64,64
        x13 = self._3t4_1(x12)  ##64,64,64,2
        x14 = self.A4D1(x13)  ##64,64,64,2

        x15 = self._4t3_1(x14)  ##32,64,64

        x17 = torch.add(x11, self.bridge1(x15))  ##64,64,64
        x18 = self._3t4_2(x17)  ## 16,64,64,4
        x19 = self.A4D2(torch.add(self.up_sample1(x14.view(x18.shape[0], -1, x18.shape[2], x18.shape[3], x18.shape[4])),x18))  ##64,64,64,4

        x19 = torch.add(F.interpolate(self.top_2(x13), size=x19.shape[2:5], mode='trilinear', align_corners=True), x19)

        x20 = self._4t3_2(x19)  ##64,64,64

        x22 = torch.add(x10, self.bridge2(
            F.interpolate(x20, size=x10.shape[2:4], mode='bilinear', align_corners=True)))  ##128,128,128
        x23 = self._3t4_3(x22)  ##16,128,128,8
        x24 = self.A4D3(
            torch.add(self.up_sample2(F.interpolate(x19, size=x23.shape[2:5], mode='trilinear', align_corners=True)),
                      x23))  ##64,128,128,8

        x24 = torch.add(F.interpolate(self.top_1(x18), size=x24.shape[2:5], mode='trilinear', align_corners=True), x24)

        x25 = self._4t3_3(x24)  ##128,128,128
        # x25 = torch.add(F.max_pool2d(x3, 4, 4),x25)
        x26 = self.tail3(x25)  ##nc,1,3

        if self.tail_add:
            x21 = self.tail2(torch.add(self.tailadd1(x25), x20))
            x16 = self.tail1(torch.add(self.tailadd2(x20), x15))
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(
                self._4t3_1.parameters()) + \
                          list(self.tail1.parameters()) + list(self.tailadd2.parameters()) + list(
                self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(
                self._4t3_2.parameters()) \
                          + list(self.tail2.parameters()) + list(self.tailadd1.parameters()) + list(
                self.A3D2.parameters())
        else:
            x16 = self.tail1(x15)  ##nc,1,3
            x21 = self.tail2(x20)  ##nc,1,3
            all_params3 = list(self._3t4_1.parameters()) + list(self.A4D1.parameters()) + list(
                self._4t3_1.parameters()) + list(self.tail1.parameters()) + list(self.A3D3.parameters())
            all_params4 = list(self.bridge1.parameters()) + list(self._3t4_2.parameters()) + \
                          list(self.A4D2.parameters()) + list(self.up_sample1.parameters()) + list(
                self._4t3_2.parameters()) + list(self.tail2.parameters()) + list(self.A3D2.parameters())

        all_params1 = list(self.head.parameters()) + list(self.layer1.parameters()) + list(self.head_l1.parameters()) + \
                      list(self.layer2.parameters()) + list(self.l1_l2.parameters()) + list(self.layer3.parameters()) + \
                      list(self.l2_l3.parameters()) + list(self.head_l3.parameters()) + list(self.layer4.parameters()) + \
                      list(self.A3D1.parameters()) + list(self.A3D2.parameters()) + list(self.A3D3.parameters())

        all_params5 = list(self.bridge2.parameters()) + list(self._3t4_3.parameters()) + list(
            self.A4D3.parameters()) + list(self.up_sample2.parameters()) \
                      + list(self._4t3_3.parameters()) + list(self.tail3.parameters()) + list(self.A3D1.parameters())

        # 计算L2正则化项
        l2_reg = []
        l2_loss = 0.0
        l2_reg.append(sum(param.norm(2) ** 2 for param in all_params1) ** 0.5)
        l2_reg.append(sum(param.norm(2) ** 2 for param in all_params3) ** 0.5)
        l2_reg.append(sum(param.norm(2) ** 2 for param in all_params4) ** 0.5)
        l2_reg.append(sum(param.norm(2) ** 2 for param in all_params5) ** 0.5)

        # st16_x = torch.unsqueeze(torch.std(x16[:, :, :, 0:x16.shape[-1] - 2], dim=(-2, -1)),dim=-1)
        # st16_y = torch.unsqueeze(torch.std(x16[:, :, :, 1:x16.shape[-1] - 1], dim=(-2, -1)),dim=-1)
        # st16_z = torch.unsqueeze(torch.std(x16[:, :, :, 2:x16.shape[-1]], dim=(-2, -1)),dim=-1)
        # st21_x = torch.unsqueeze(torch.std(x21[:, :, :, 0:x21.shape[-1] - 2], dim=(-2, -1)),dim=-1)
        # st21_y = torch.unsqueeze(torch.std(x21[:, :, :, 1:x21.shape[-1] - 1], dim=(-2, -1)),dim=-1)
        # st21_z = torch.unsqueeze(torch.std(x21[:, :, :, 2:x21.shape[-1]], dim=(-2, -1)),dim=-1)
        # st26_x = torch.unsqueeze(torch.std(x26[:, :, :, 0:x26.shape[-1] - 2], dim=(-2, -1)),dim=-1)
        # st26_y = torch.unsqueeze(torch.std(x26[:, :, :, 1:x26.shape[-1] - 1], dim=(-2, -1)),dim=-1)
        # st26_z = torch.unsqueeze(torch.std(x26[:, :, :, 2:x26.shape[-1]], dim=(-2, -1)),dim=-1)
        # uncertain16 = torch.unsqueeze(torch.cat((torch.cat([st16_x,st16_y],dim=-1),st16_z),dim=-1),dim=2)
        # uncertain21 = torch.unsqueeze(torch.cat((torch.cat([st21_x,st21_y],dim=-1),st21_z),dim=-1),dim=2)
        # uncertain26 = torch.unsqueeze(torch.cat((torch.cat([st26_x,st26_y],dim=-1),st26_z),dim=-1),dim=2)
        # uncertain =[uncertain16,uncertain21,uncertain26]

        x16 = self.adv(x16)   #view(x16.shape[0],self.num_classes,1,3)
        x21 = self.adv(x21)
        x26 = self.adv(x26)

        n0 = (x26[:, :, :, 0] + x21[:, :, :, 0] + x16[:, :, :, 0]) / 3.
        n1 = (x26[:, :, :, 1] + x21[:, :, :, 1] + x16[:, :, :, 1]) / 3.
        n2 = (x26[:, :, :, 2] + x21[:, :, :, 2] + x16[:, :, :, 2]) / 3.
        m0 = (((x16[:, :, :, 0] - n0)**2 + (x26[:, :, :, 0] - n0)**2 + (x21[:, :, :, 0] - n0)**2)/3.)**0.5
        m1 = (((x16[:, :, :, 1] - n1) ** 2 + (x26[:, :, :, 1] - n1) ** 2 + (x21[:, :, :, 1] - n1) ** 2) / 3.) ** 0.5
        m2 = (((x16[:, :, :, 2] - n2) ** 2 + (x26[:, :, :, 2] - n2) ** 2 + (x21[:, :, :, 2] - n2) ** 2) / 3.) ** 0.5

        #xnorm = torch.unsqueeze(torch.cat(((torch.cat((((x26[:,:,:,0]-x21[:,:,:,0])**2 + (x26[:,:,:,0]-x16[:,:,:,0])**2)**0.5,((x26[:,:,:,1]-x21[:,:,:,1])**2 + (x26[:,:,:,1]-x16[:,:,:,1])**2)**0.5),dim=-1)
        #                    ,((x26[:,:,:,2]-x21[:,:,:,2])**2 + (x26[:,:,:,2]-x16[:,:,:,2])**2)**0.5)),dim=-1),dim=2)
        xnorm = torch.unsqueeze(torch.cat((torch.cat((m0,m1),dim=-1),m2),dim=-1),dim=2)

        for i in l2_reg:
            l2_loss += (i/sum(l2_reg)) * i

        if setting_config.relevant:
            xyz1 = torch.squeeze(x16, dim=2)
            x16T = torch.unsqueeze(self.Relevance_adjustments(xyz1,xnorm), dim=2)
            xyz2 = torch.squeeze(x21, dim=2)
            x21T = torch.unsqueeze(self.Relevance_adjustments(xyz2,xnorm), dim=2)
            xyz = torch.squeeze(x26, dim=2)
            x26T = torch.unsqueeze(self.Relevance_adjustments(xyz,xnorm), dim=2)
        else:
            x16T, x21T, x26T = x16,x21,x26
        T = [x16,x21,x26]

        if self.deepsuper:
            return [l2_loss,[T, x16T], [xnorm, x21T]], x26T
        else:
            return x26

    def Relevance_adjustments(self,xyz,xnorm):
        # a1_opt=torch.tensor([[3.14234379e-06 , 2.60536160e-06 ,-1.05595981e-05]])
        # a2_opt=torch.tensor([[5.92477343e-06 ,-2.02061947e-06 ,-6.01353748e-07]])
        # a3_opt=torch.tensor([[-3.44045876e-08 ,-1.39870307e-06 , 7.20317632e-06]])
        # a4_opt=torch.tensor([[-4.78598119e-06 , 1.77299450e-06 , 1.66281140e-07]])
        # a5_opt=torch.tensor([[-4.29633469e-06 ,-4.77312140e-06 , 3.43732168e-06]])
        # a6_opt=torch.tensor([[-2.52408476e-07 , 3.52046535e-06 , 1.41244982e-06]])
        # b = torch.tensor(0.00022636).cuda()

        a1_opt=torch.tensor([[-6.21007172e-05 , 1.28758479e-06 ,-2.68045118e-04]])
        a2_opt=torch.tensor([[-2.07947256e-04 ,4.46834409e-05 , 6.14944605e-06]])
        a3_opt=torch.tensor([[-2.21353520e-04 , 6.30610806e-05 ,1.87035802e-04]])
        a4_opt=torch.tensor([[2.54301043e-04, -9.25418365e-05 ,-7.99169087e-06]])
        a5_opt=torch.tensor([[2.42138186e-04  ,1.33552915e-04  ,4.71636951e-05]])
        a6_opt=torch.tensor([[4.30034321e-06,-1.38483871e-04,  7.74779972e-05]])
        #b = torch.tensor(3.88585612e-03).cuda()
        b = torch.tensor(2).cuda()

        mean1 = [85.38860869565218, 42.480782608695655 ,100.6086956521739]
        std1 = [19.910707448700823, 9.613875221721361, 8.090462073302508]
        mean2 = [59.140608695652176, 41.85843478260869 ,101.49565217391304]
        std2 = [19.193310124690143, 10.816812645644006 ,8.226546544749507]
        mean3 = [59.077999999999996 ,42.2328695652174, 101.6]
        std3 = [19.56582149521478, 10.853970843426072 ,8.433679892060116]
        mean4 = [21.51991304347826, 58.56895652173912 ,99.77391304347826]
        std4 = [17.483606109757055, 15.138758054506724, 8.966663738600563]
        mean5 = [140.95017391304347 ,37.60078260869565 ,101.03478260869565]
        std5 = [21.507908918737563, 10.001301232541962, 8.219732815696423]
        mean6 = [137.40017391304346, 101.50678260869566 ,99.73913043478261]
        std6 = [19.25242024232656 ,10.279309530848941 ,3.934434286768532]

        mean=torch.tensor([mean1,mean2,mean3,mean4,mean5,mean6]).cuda()
        std=torch.tensor([std1,std2,std3,std4,std5,std6]).cuda()
        a = torch.unsqueeze(torch.cat((a1_opt, a2_opt, a3_opt, a4_opt, a5_opt, a6_opt),dim=0),dim=0).cuda()

        relev_loss = torch.zeros(xyz.shape[0]).cuda()
        new_xyz = torch.zeros(xyz.shape)
        for kd,data in enumerate(xyz):
            y = torch.sum(torch.sum(a * (data**2), axis=2))-b
            last_y=100000
            last_data = data.cuda()
            for i in range(20):
                for j,jp in enumerate(a[0]):
                    dot = 2 * y * jp.cuda() * xnorm[kd,j,0,:] * data[j]
                    new_data = data[j] - dot
                    new_xyz[kd,j] = new_data
                new_y = torch.sum(torch.sum(a * (new_xyz[kd].cuda()**2), axis=2)) - b
                # print("%d,lossdd===%f"%(i,new_y))
                if abs(new_y)>=abs(last_y):
                    new_xyz[kd] = last_data
                    break
                else:
                    data = new_xyz[kd].cuda()
                    last_data = new_xyz[kd].cuda()
                    last_y = new_y
                    y = new_y

            relev_loss[kd] = new_y
        # dis = torch.unsqueeze(relev_loss,dim=1)
        #print("lastdata=",xyz,"newdata=",new_xyz)
        return new_xyz.cuda()