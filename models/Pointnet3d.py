import torch
import torch.nn as nn

# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, input_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv3d(input_channels // 4, input_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attention = self.conv2(nn.ReLU()(self.conv1(x)))
        attention = self.softmax(attention)
        out = x * attention
        return out


# 定义3D深度可分离特征提取模块
class My3DDepthwiseSeparableNet(nn.Module):
    def __init__(self,n_classes):
        super(My3DDepthwiseSeparableNet, self).__init__()
        self.conv1 = nn.Conv3d(n_classes, 3, kernel_size=3, stride=1, padding=1, groups=3)  # 3D深度可分离卷积
        # self.attention = AttentionModule(3)
        self.conv2 = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.tail1 = nn.Conv3d(32, 24, kernel_size=3, stride=1, padding=1)
        self.tail2 = nn.Conv3d(24,n_classes, kernel_size = 3, stride = 1, padding = 1)
        self.avpool = nn.AdaptiveAvgPool2d((1, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        # x = self.attention(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.tail1(x)
        x = nn.ReLU()(x)
        x = self.tail2(x)
        x = self.avpool(x)

        return x

# 定义多尺度特征融合模块
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1x1 = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.conv3x3 = nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv3d(input_channels, output_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x3 = self.conv3x3(x)
        x5 = self.conv5x5(x)
        out = torch.cat((x1, x3, x5), dim=1)
        return out

class pointnet3d(nn.Module):
    def __init__(self,n_classes=3,n_channels=49):
        super().__init__()
        self.layer=nn.Sequential(
            MultiScaleFeatureFusion(n_channels,n_classes),
            My3DDepthwiseSeparableNet(n_classes)
    )

    def forward(self,x):
        x=self.layer(x)

        return x

