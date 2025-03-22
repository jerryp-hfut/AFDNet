import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward

class WaveletAttentionModule(nn.Module):
    def __init__(self, in_channels, wave='haar', reduction=16):
        super(WaveletAttentionModule, self).__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave=wave)
        self.in_channels = in_channels

        # 确保通道数至少为1
        reduced_channels = max(in_channels // reduction, 1)

        # 条状卷积用于水平和垂直分量
        self.horizontal_attention = self.create_horizontal_attention(in_channels, reduction)
        self.vertical_attention = self.create_vertical_attention(in_channels, reduction)

        # 对角线分量的常规卷积
        self.diagonal_attention = self.create_diagonal_attention(in_channels, reduction)

        # 低频分量的全局通道注意力机制
        self.low_freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def create_horizontal_attention(self, in_channels, reduction):
        """水平高频分量的条状卷积（横向卷积）"""
        reduced_channels = max(in_channels // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=(1, 5), padding=(0, 2)),  # 横向条状卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=(1, 5), padding=(0, 2)),  # 再次横向卷积
            nn.Sigmoid()
        )

    def create_vertical_attention(self, in_channels, reduction):
        """垂直高频分量的条状卷积（纵向卷积）"""
        reduced_channels = max(in_channels // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=(5, 1), padding=(2, 0)),  # 纵向条状卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),  # 再次纵向卷积
            nn.Sigmoid()
        )

    def create_diagonal_attention(self, in_channels, reduction):
        """对角线高频分量的标准卷积"""
        reduced_channels = max(in_channels // reduction, 1)
        return nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=3, padding=1),  # 标准卷积
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=3, padding=1),  # 标准卷积
            nn.Sigmoid()
        )

    def forward(self, x):
        # 小波变换
        yL, yH = self.dwt(x)

        # 处理低频分量
        low_freq = yL
        low_freq_attn = self.low_freq_attention(low_freq)
        low_freq = low_freq * low_freq_attn

        # 处理高频分量
        high_freq_h = yH[0][:, :, 0, :, :]  # 水平分量
        high_freq_v = yH[0][:, :, 1, :, :]  # 垂直分量
        high_freq_d = yH[0][:, :, 2, :, :]  # 对角线分量

        # 应用不同的注意力机制
        high_freq_h = high_freq_h * self.horizontal_attention(high_freq_h)
        high_freq_v = high_freq_v * self.vertical_attention(high_freq_v)
        high_freq_d = high_freq_d * self.diagonal_attention(high_freq_d)

        # 合并高频分量
        high_freq = torch.cat([high_freq_h, high_freq_v, high_freq_d], dim=1)

        # 特征融合，低频 + 高频
        combined = torch.cat([low_freq, high_freq], dim=1)
        out = self.fusion_conv(combined)

        return out

class RainDropFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 初始化父类nn.Module
        super(RainDropFeatureExtractor, self).__init__()
        # 创建WaveletAttentionModule实例，用于对输入特征进行小波变换和注意力机制处理
        self.wavelet_attention = WaveletAttentionModule(in_channels)
        # 创建一个卷积层序列，包括一个2D卷积层和一个ReLU激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 2D卷积层，输入通道数为in_channels，输出通道数为out_channels，卷积核大小为3x3，padding为1
            nn.ReLU(inplace=True)  # ReLU激活函数，inplace=True表示在原地进行操作，节省内存
        )
    def forward(self, x):
        # 通过WaveletAttentionModule对输入特征进行处理，得到注意力机制处理后的特征
        attention_out = self.wavelet_attention(x)
        # 将注意力机制处理后的特征输入到卷积层序列中，得到卷积后的特征
        conv_out = self.conv(attention_out)
        # 输出卷积后的特征
        out = conv_out
        return out, attention_out

def bilinear_interpolate(input, grid):
    # grid 的范围是 [-1, 1]，这里我们假设输入是 BxCxHxW 的形状
    return F.grid_sample(input, grid, align_corners=True)

# 升级后的 Deformable ConvTranspose 实现，带残差连接
class DeformConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, bias=True):
        super(DeformConvTranspose2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # 标准反卷积层
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

        # 生成偏移量的卷积层（加入多尺度偏移）
        self.offset_conv1 = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)
        self.offset_conv2 = nn.Conv2d(in_channels, 2, kernel_size=5, padding=2)

        # 使用 1x1 卷积融合多尺度偏移
        self.offset_fuse = nn.Conv2d(4, 2, kernel_size=1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用 kaiming_normal 初始化卷积层权重
        nn.init.kaiming_normal_(self.conv_transpose.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.offset_conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.offset_conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.offset_fuse.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 生成多尺度偏移量
        offset1 = self.offset_conv1(x) # 偏移量1
        offset2 = self.offset_conv2(x)  # 偏移量2
        
        # 多尺度偏移融合
        offset = torch.cat([offset1, offset2], dim=1)
        offset = torch.tanh(self.offset_fuse(offset))

        # 生成初始反卷积特征图
        out = self.conv_transpose(x)
        res = out
        # 获取输出特征图的大小
        n, c, h, w = out.shape
        
        # 上采样偏移量，使其与反卷积输出的 h 和 w 一致
        offset = F.interpolate(offset, size=(h, w), mode='bilinear', align_corners=True)

        # 生成常规的网格坐标
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid = torch.stack((grid_x, grid_y), dim=-1).float()  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1).to(x.device)  # 扩展维度 [N, H, W, 2]

        # 将偏移量 reshape 成与采样点匹配的形状
        offset = offset.permute(0, 2, 3, 1)  # [N, H, W, 2]

        # 添加偏移量到常规采样点上，构成新的采样网格
        grid = grid + offset  # [N, H, W, 2]

        # 将采样网格归一化到 [-1, 1] 的范围（grid_sample 的要求）
        grid = (grid / torch.tensor([w-1, h-1]).to(x.device) - 0.5) * 2

        # 使用双线性插值对新的网格点进行采样
        out = bilinear_interpolate(out, grid)
        
        # 引入残差连接，将输入与输出相加
        out = out + res
        
        return out

class DynamicReLUIdentity2d(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channels = channels
        
        # 通道级动态权重
        self.channel_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),  # 降维
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),  # 升维
            nn.Sigmoid()  # 权重限制在 [0, 1]
        )
        
        # 空间感知的动态权重
        self.spatial_weight = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),  # 降维
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, 1, kernel_size=1),  # 输出单通道权重图
            nn.Sigmoid()  # 权重限制在 [0, 1]
        )
        
        # 归一化层
        self.norm = nn.GroupNorm(num_groups=16, num_channels=channels)  # GroupNorm 更适合图像任务

    def forward(self, x):
        x_norm = self.norm(x)  # 归一化输入
        
        # 通道级动态权重
        channel_weight = self.channel_weight(x_norm)  # [B, C, 1, 1]
        # 空间感知的动态权重
        spatial_weight = self.spatial_weight(x_norm)  # [B, 1, H, W]
        # 综合权重
        weight = channel_weight * spatial_weight  # [B, C, H, W]
        # 动态混合 ReLU 和恒等映射
        relu_out = F.relu(x)
        identity_out = x
        output = weight * relu_out + (1 - weight) * identity_out
        
        return output
        
class AdaptivePConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, partial_ratio=0.25, stride=1, padding=1, bias=True):
        super(AdaptivePConv, self).__init__()
        
        self.in_channels = in_channels
        self.partial_channels = int(in_channels * partial_ratio)
        
        # 卷积层
        self.conv = nn.Conv2d(self.partial_channels, out_channels, 
                             kernel_size=kernel_size, 
                             stride=stride, 
                             padding=padding, 
                             bias=bias)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算每个通道的活跃度 (使用标准差作为度量)
        channel_activity = torch.std(x, dim=(2, 3))  # [B, C]
        
        # 选择活跃度最高的 partial_channels 个通道的索引
        _, top_indices = torch.topk(channel_activity, self.partial_channels, dim=1)
        
        # 为每个批次样本收集重要通道
        partial_outputs = []
        untouched_outputs = []
        
        for i in range(batch_size):
            # 获取当前样本的活跃通道
            selected_channels = x[i, top_indices[i], :, :]
            selected_channels = selected_channels.unsqueeze(0)
            
            # 对选中的通道进行卷积
            partial_out = self.conv(selected_channels)
            partial_outputs.append(partial_out)
            
            # 获取未选中的通道
            mask = torch.ones(self.in_channels, device=x.device)
            mask[top_indices[i]] = 0
            unselected_indices = mask.nonzero().squeeze()
            untouched_channels = x[i, unselected_indices, :, :]
            untouched_channels = untouched_channels.unsqueeze(0)
            
            untouched_outputs.append(untouched_channels)
        
        # 将所有批次的输出组合
        partial_output = torch.cat(partial_outputs, dim=0)
        untouched_output = torch.cat(untouched_outputs, dim=0)
        
        # 拼接处理过的和未处理的通道
        out = torch.cat((partial_output, untouched_output), dim=1)
        
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积：在每个通道上单独应用卷积，不跨通道
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=in_channels, bias=bias)
        # 逐点卷积：1x1 卷积用于通道之间的线性组合
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        # 深度卷积
        x = self.depthwise(x)
        # 逐点卷积
        x = self.pointwise(x)
        return x

class WDNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(WDNet, self).__init__()

        # Encoder
        self.encoder1 = RainDropFeatureExtractor(in_channels, 64)
        self.encoder2 = RainDropFeatureExtractor(64, 128)
        self.encoder3 = RainDropFeatureExtractor(128, 256)
        self.encoder4 = RainDropFeatureExtractor(256, 512)

        # Middle part
        
        self.middle = nn.Sequential(
            DepthwiseSeparableConv(in_channels=512, out_channels=1024),
            AdaptivePConv(1024, 512),
            DynamicReLUIdentity2d(1280),
            DepthwiseSeparableConv(in_channels=1280, out_channels=512),
        )
        
        # Decoder
        self.upconv4 = DeformConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = self.double_conv(256 + 256, 256)
        self.upconv3 = DeformConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = self.double_conv(128 + 128, 128)
        self.upconv2 = DeformConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = self.double_conv(64 + 64, 64)
        self.upconv1 = DeformConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder1 = self.double_conv(32 + 3, 64)
        self.up = DeformConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        res = x
        # Encoder
        x1, skip1 = self.encoder1(x)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)
        # Middle part
        x5 = self.middle(self.maxpool(x4))
        # Decoder
        x = self.upconv4(x5)
        x = F.interpolate(x, size=skip4.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip4], dim=1)
        x = self.decoder4(x)
        x = self.upconv3(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder1(x)
        # Output layer
        x = self.up(x)
        x = self.out_conv(x)
        x = x + res
        x = self.final_conv(x)
        return x

    def maxpool(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

if __name__ == '__main__':
    model = WDNet(in_channels=3, out_channels=3)
    input_tensor = torch.rand(1, 3, 480, 720)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)