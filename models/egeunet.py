import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
import math

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        '''
        这是一个depthwise卷积层,保持输入输出通道数相同,常用于保持空间分辨率不变的情况。
        这个conv1层不改变输入tensor的channel大小,而是混合channel之间的信息,起到一个信息交互的作用。
        '''
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation,
                               groups=dim_in)  ##dilation: 膨胀或空洞系数,默认是1;  groups: 分组卷积,设置为dim_in即为depthwise卷积
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

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


class group_aggregation_bridge(nn.Module):  ##GAB
    '''
    这种多分支设计可以获得较大感受野,同时保持计算效率。通过组卷积和膨胀卷积的组合使用,可以有效地实现多尺度特征融合。
    '''

    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1, 2, 5, 7], gt_ds=True):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        gs = group_size + 1 if gt_ds else group_size
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=gs, data_format='channels_first'),
            nn.Conv2d(gs, gs, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=gs)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=gs, data_format='channels_first'),
            nn.Conv2d(gs, gs, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=gs)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=gs, data_format='channels_first'),
            nn.Conv2d(gs, gs, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=gs)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=gs, data_format='channels_first'),
            nn.Conv2d(gs, gs, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=gs)
        )
        if gt_ds:
            self.tail_conv = nn.Sequential(
                LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
                nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
            )
        else:
            self.tail_conv = nn.Sequential(
                LayerNorm(normalized_shape=dim_xl * 2, data_format='channels_first'),
                nn.Conv2d(dim_xl * 2, dim_xl, 1))

    def forward(self, xh, xl, mask=[]):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear',
                           align_corners=True)  ##上采样，双线性插值法 GAB是输入到decode中融合
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)  # 对第一位维度即通道进行切分

        if len(mask):
            x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
            x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
            x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
            x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        else:
            x0 = self.g0(torch.cat((xh[0], xl[0]), dim=1))
            x1 = self.g1(torch.cat((xh[1], xl[1]), dim=1))
            x2 = self.g2(torch.cat((xh[2], xl[2]), dim=1))
            x3 = self.g3(torch.cat((xh[3], xl[3]), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


'''
为了克服MHSA提出的二次复杂度问题，我们提出了具有线性复杂度的HPA。
给定输入x和随机初始化的可学习张量p，首先使用双线性插值来调整p的大小以匹配x的大小。
然后，我们对p使用深度可分离卷积（DW），然后在x和p之间进行阿达玛乘积运算以获得输出。
然而，单独使用简单的HPA不足以从多个角度提取信息，导致结果不令人满意。受MHSA中多头模式的启发，我们引入了基于HPA的GHPA，如下算法所示。
我们沿着通道维度将输入等分为四组，并分别在前三组的高度-宽度、通道高度和通道宽度轴上执行HPA。对于最后一组，我们只在特征图上使用DW。
最后，我们沿着通道维度将四个组连接起来，并应用另一个DW来整合来自不同角度的信息。注意，DW中使用的所有内核大小都是3。
'''


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):  # GHPA
    def __init__(self, dim_in, dim_out, x=8, y=8):  # 初始化函数，定义了输入的维度 dim_in、输出的维度 dim_out，以及默认的 x 和 y。
        super().__init__()  # 调用父类的初始化函数。

        c_dim_in = dim_in // 4  # 计算输入维度的四分之一，用于后续的分组卷积操作。
        k_size = 3  # 设置卷积核的大小为3。
        pad = (k_size - 1) // 2  # 计算卷积操作的padding大小，以保持输入输出的尺寸一致。

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y),
                                      requires_grad=True)  # p：定义一个可学习的参数 params_xy，它是一个具有大小为 (1, c_dim_in, x, y) 的四维张量。
        nn.init.ones_(self.params_xy)  # 对参数 params_xy 进行初始化，将其填充为全1的张量。
        # 定义一个卷积层 conv_xy，其中包含多个操作：分组卷积操作、GELU激活函数以及再次的卷积操作。
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x),
                                      requires_grad=True)  # 定义一个可学习的参数 params_zx，它是一个具有大小为 (1, 1, c_dim_in, x) 的四维张量。
        nn.init.ones_(self.params_zx)  # 对参数 params_zx 进行初始化，将其填充为全1的张量。后类似---
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )  ##定义一个深度卷积层 dw，其中包含多个操作：1x1卷积、GELU激活函数以及3x3分组卷积操作。深度可分离卷积的参数数量较少，从而减少了模型的复杂度。

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')  # 定义一个标准化层norm1，用于对输入进行归一化处理。
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')  # 定义一个标准化层 norm2，用于对特征图进行归一化处理。

        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )  # 定义最后的低维卷积层 ldw，用于将特征图转换为最终的输出维度。

    def forward(self, x):  # 定义前向传播函数，接收输入 x，并返回模型的输出。
        x = self.norm1(x)  # 对输入进行归一化处理。
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)  # 将输入按通道维度进行分块，得到四个块：x1、x2、x3和x4
        B, C, H, W = x1.size()  # 获取分块后 x1 的维度信息。
        # ----------xy----------#
        params_xy = self.params_xy  # 获取参数 params_xy。(1,C,X,Y)->(1,C,W,H)
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear',
                                             align_corners=True))  ##上采样用的双线性插值，变尺寸与x1大小一致，对 x1 进行卷积操作，并与 params_xy 进行元素逐元素相乘。
        # ----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)  # (N,C,W,H)->(N,H,C,W)   对 x2 进行维度重排，将通道维度移动到前面。将tensor调整为Conv1d层需要的输入格式
        params_zx = self.params_zx  ###########(1,1,C,X)->(1,1,C,W)->(1,C,W)->(1,1,C,W)
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(
            0)  # squeeze,unsqueeze去掉并加回batch维，align_corners=True表示在采样时保留角点像素的位置。
        x2 = x2.permute(0, 2, 3, 1)  # 对 x2 进行维度重排，将通道维度移动到最后面。(N,H,C,W)->(N,C,W,H)
        # ----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)  # 对 x3 进行维度重排，交换通道维度与横向维度。(N,C,W,H)->(N,W,C,H)
        params_zy = self.params_zy  ###(1,1,C,Y)->(1,1,C,H)->(1,C,H)->(1,1,C,H)
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)  # 对 x3 进行维度重排，交换通道维度与横向维度。(N,C,W,H)->(N,C,W,H)

        # ----------dw----------#
        x4 = self.dw(x4)  # 对 x4 进行深度卷积操作。
        # ----------concat----------#
        x = torch.cat([x1, x2, x3, x4], dim=1)  # 将 x1、x2、x3和x4 沿通道维度拼接起来，形成一个新的特征图 x。
        # ----------ldw----------#
        x = self.norm2(x)  # 对特征图进行归一化处理。
        x = self.ldw(x)  # 对特征图进行低维卷积操作，得到最终的输出。
        return x


class EGEUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], bridge=True, gt_ds=True,
                 deep_supervision=False):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        self.deep_supervision = deep_supervision
        self.w = nn.Parameter(torch.FloatTensor([-2.5, -2, -1.5, -1, -0.5]))


        self.avgpool=nn.AdaptiveAvgPool2d((1, 3))

        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        if bridge:
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0], gt_ds=gt_ds)
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1], gt_ds=gt_ds)
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2], gt_ds=gt_ds)
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3], gt_ds=gt_ds)
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4], gt_ds=gt_ds)
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')

        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        )
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        )
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

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

    def forward(self, x):

        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out  # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out  # b, c1, H/4, W/4

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out  # b, c2, H/8, W/8

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out  # b, c3, H/16, W/16

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        t5 = out  # b, c4, H/32, W/32

        out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        t6 = out

        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        if self.bridge:
            if self.gt_ds:
                gt_pre5 = self.gt_conv1(out5)
                t5 = self.GAB5(t6, t5, gt_pre5)
                #gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)
                gt_pre5=self.avgpool(gt_pre5)
            else:
                t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        if self.bridge:
            if self.gt_ds:
                gt_pre4 = self.gt_conv2(out4)
                t4 = self.GAB4(t5, t4, gt_pre4)
                #gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
                gt_pre4 = self.avgpool(gt_pre4)
            else:
                t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        if self.bridge:
            if self.gt_ds:
                gt_pre3 = self.gt_conv3(out3)
                t3 = self.GAB3(t4, t3, gt_pre3)
                #gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
                gt_pre3 = self.avgpool(gt_pre3)
            else:
                t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        if self.bridge:
            if self.gt_ds:
                gt_pre2 = self.gt_conv4(out2)
                t2 = self.GAB2(t3, t2, gt_pre2)
                #gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
                gt_pre2 = self.avgpool(gt_pre2)
            else:
                t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        if self.bridge:
            if self.gt_ds:
                gt_pre1 = self.gt_conv5(out1)
                t1 = self.GAB1(t2, t1, gt_pre1)
                #gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
                gt_pre1 = self.avgpool(gt_pre1)
            else:
                t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        #out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',align_corners=True)  # b, num_class, H, W
        out0 = self.avgpool(self.final(out1))

        w = torch.sigmoid(self.w)
        if self.gt_ds and self.deep_supervision:
            return [[w[0],gt_pre5], [w[1],gt_pre4],[w[2],gt_pre3],[w[3],gt_pre2],[w[4],gt_pre1]], out0
        else:
            return out0