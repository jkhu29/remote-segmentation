from typing import List
import torch
from torch._C import StringType
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import backbone


def _make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ConvReLU(nn.Module):
    """docstring for ConvReLU"""

    def __init__(self, channels: int = 2048, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1,
                 out_channels=None):
        super(ConvReLU, self).__init__()
        if out_channels is None:
            self.out_channels = channels
        else:
            self.out_channels = out_channels

        self.conv = nn.Conv2d(channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              bias=False)
        self.bn = nn.GroupNorm(self.out_channels, self.out_channels)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# 2048 --> 256 赋予每个通道的权值
class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 2048, reduction: int = 8):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.CELU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        score = self.conv(self.avg_pool(x))
        return x * score


# 2048 --> 256
class PositionAttention(nn.Module):
    """docstring for PositionAttention"""

    def __init__(self, channels: int = 2048, reduction: int = 8):
        super(PositionAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv2 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, c, h, w = x.shape
        # n --> batch_size
        # c --> channel
        # h --> height
        # w --> width
        # permute 换位函数
        # 4维张量转换为3维数据，交换位置满足矩阵相乘，为什么相乘 --> 空间中的相似度越高向量乘积越大
        # x_embed1: B卷积  x_embed2: C卷积
        x_embed1 = self.conv1(x).view(n, -1, h * w).permute(0, 2, 1)
        x_embed2 = self.conv2(x).view(n, -1, h * w)
        # attention: S
        attention = F.softmax(torch.bmm(x_embed1, x_embed2), dim=-1)
        # x_embed1: D卷积
        x_embed1 = self.conv3(x).view(n, -1, h * w)
        x_embed2 = torch.bmm(x_embed1, attention.permute(0, 2, 1)).view(n, -1, h, w)
        x = self.alpha * x_embed2 + x

        return x


class NonLocalAttention(nn.Module):
    def __init__(self, channels: int = 2048):
        super(NonLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, 1)
        self.relu1 = nn.CELU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, 1, 1)
        self.relu2 = nn.CELU(inplace=True)

        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.relu3 = nn.CELU(inplace=True)

    def forward(self, x):
        x_embed1 = self.relu1(self.conv1(x))
        x_embed2 = self.relu2(self.conv2(x))
        x_assembly = self.relu3(self.conv3(x))

        n, c, h, w = x_embed1.shape
        x_embed1 = x_embed1.permute(0, 2, 3, 1).view(n, h * w, c)
        x_embed2 = x_embed2.view(n, c, h * w)
        score = torch.matmul(x_embed1, x_embed2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(n, -1, h * w).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly).permute(0, 2, 1).view(n, -1, h, w)

        return x_final


class NonLocalPositionAttention(nn.Module):
    def __init__(self, channels: int = 2048, reduction: int = 8):
        super(NonLocalPositionAttention, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels // reduction, 1, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.relu3 = nn.ReLU(inplace=True)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        n, _, h, w = x.shape
        x_embed1 = self.relu1(self.conv1(x)).view(n, -1, h * w).permute(0, 2, 1)
        x_embed2 = self.relu2(self.conv2(x)).view(n, -1, h * w)
        x_assembly = self.relu3(self.conv3(x))

        attention = F.softmax(torch.bmm(x_embed1, x_embed2), dim=-1)
        x_embed2 = torch.bmm(x.view(n, -1, h * w), attention.permute(0, 2, 1)).view(n, -1, h, w)

        return self.alpha * x_embed2 + x_assembly


class ASPP(nn.Module):
    def __init__(self, channels: int = 2048, out_channels: int = 256, kernel_size: int = 3,
                 dilation_rate: List = [8, 12, 16]):
        super(ASPP, self).__init__()
        self.conv0 = ConvReLU(channels=channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.conv1 = ConvReLU(channels=channels, out_channels=out_channels, dilation=1)
        self.conv2 = ConvReLU(channels=channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=dilation_rate[0], dilation=dilation_rate[0])
        self.conv3 = ConvReLU(channels=channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=dilation_rate[1], dilation=dilation_rate[1])
        self.conv4 = ConvReLU(channels=channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=kernel_size, dilation=dilation_rate[2])
        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvReLU(channels=channels, out_channels=out_channels, kernel_size=1),
        )
        self.tail = ConvReLU(channels=out_channels * 5, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.shape
        size = (h, w)
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = F.interpolate(self.conv4(x), size=size, mode="bilinear", align_corners=False)
        res = torch.cat([x0, x1, x2, x3, x4], dim=1)
        return self.tail(res)


class DeepLabV3(nn.Module):
    def __init__(self, channels: int = 2048, out_channels: int = 256, dilation_rate: List = [8, 12, 16]):
        super(DeepLabV3, self).__init__()
        self.aspp1 = ConvReLU(channels=channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.aspp2 = ConvReLU(channels=channels, out_channels=out_channels, dilation=1)
        self.aspp3 = ASPP(channels=channels, out_channels=out_channels, dilation_rate=dilation_rate)
        self.aspp4 = ASPP(channels=channels, out_channels=out_channels, dilation_rate=dilation_rate)
        self.tail = ConvReLU(channels=out_channels * 4, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.shape
        size = (h, w)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        res = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.tail(res)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        return x


class DAHead(nn.Module):
    """docstring for DAHead"""

    def __init__(self, n_classes: int, channels: int = 2048, reduction: int = 8, aux: bool = False):
        super(DAHead, self).__init__()
        self.aux = aux
        self.conv_p1 = _make_layer(ConvReLU, num_layers=1)
        self.conv_p2 = _make_layer(ConvReLU, num_layers=1)
        self.conv_n1 = _make_layer(ConvReLU, num_layers=1)
        self.conv_n2 = _make_layer(ConvReLU, num_layers=1)
        self.conv_c1 = _make_layer(ConvReLU, num_layers=1)
        self.conv_c2 = _make_layer(ConvReLU, num_layers=1)
        self.pa = PositionAttention(channels=channels)
        self.ca = ChannelAttention(channels=channels)
        self.nla = NonLocalAttention(channels=channels)
        self.conv1 = ConvReLU(channels=channels * 3, out_channels=channels, kernel_size=1, padding=0)
        self.out = DeepLabV3(channels, channels // reduction)
        self.conv2 = nn.Conv2d(channels // reduction, n_classes, 3, 1, 1, bias=True)
        if self.aux:
            self.conv_p3 = nn.Conv2d(channels, n_classes, 1)
            self.conv_c3 = nn.Conv2d(channels, n_classes, 1)
            self.conv_n3 = nn.Conv2d(channels, n_classes, 1)

    def forward(self, x):
        x1 = self.conv_p1(x)
        x1 = self.pa(x1)
        x1 = self.conv_p2(x1)

        x2 = self.conv_c1(x)
        x2 = self.ca(x2)
        x2 = self.conv_c2(x2)

        x3 = self.conv_n1(x)
        x3 = self.nla(x3)
        x3 = self.conv_n2(x3)

        fusion = torch.cat([x1, x2, x3], 1)
        fusion = self.conv1(fusion)
        out = self.conv2(self.out(fusion))
        output = [out]
        if self.aux:
            out1 = self.conv_p3(x1)
            out2 = self.conv_c3(x2)
            out3 = self.conv_n3(x3)
            output.append(out1)
            output.append(out2)
            output.append(out3)
        return tuple(output)


class DANet(nn.Module):
    """docstring for DANet"""
    def __init__(self, channels: int = 256, aux: bool = True):
        super(DANet, self).__init__()
        self.aux = aux
        self.resnext = backbone.resnext50_32x4d()
        pretrained_resnext = torchvision.models.resnext50_32x4d(pretrained=True)
        pretrained_dict = pretrained_resnext.state_dict()
        model_dict = self.resnext.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.resnext.load_state_dict(model_dict)
        for params in self.resnext.parameters():
            params.requires_grad = False

        self.dahead = DAHead(channels, aux=self.aux)

    def forward(self, x):
        _, _, h, w = x.shape
        feature_map = self.resnext(x)

        x = self.dahead(feature_map)
        x0 = F.interpolate(x[0], (h, w), mode='bilinear', align_corners=True)
        output = [x0]

        if self.aux:
            x1 = F.interpolate(x[1], (h, w), mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], (h, w), mode='bilinear', align_corners=True)
            x3 = F.interpolate(x[3], (h, w), mode='bilinear', align_corners=True)
            output.append(x1)
            output.append(x2)
            output.append(x3)
        return tuple(output)


class ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels: int = 256, key_channels: int = 128, scale: int = 1):
        super(ObjectAttentionBlock, self).__init__()
        self.key_channels = key_channels
        self.scale = scale
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

        self.to_q = nn.Sequential(
            ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0),
            ConvReLU(key_channels, out_channels=key_channels, kernel_size=1, padding=0)
        )
        self.to_k = nn.Sequential(
            ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0),
            ConvReLU(key_channels, out_channels=key_channels, kernel_size=1, padding=0)
        )
        self.to_v = ConvReLU(in_channels, out_channels=key_channels, kernel_size=1, padding=0)
        self.f_up = ConvReLU(key_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, features, context):
        n, c, h, w = features.shape
        if self.scale == 1:
            features = self.pool(features)

        query = self.to_q(features).view(n, -1, c)
        key = self.to_k(context).view(n, c, -1)
        value = self.to_v(context).view(n, -1, c)

        sim_map = torch.matmul(query, key)
        sim_map *= self.key_channels ** -0.5
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map.squeeze(), value.squeeze()).view(n, c, h, w)

        context = self.f_up(context)
        context = self.up(context)
        return context


class SpatialOCR(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.5):
        super(SpatialOCR, self).__init__()
        self.object_context_block = ObjectAttentionBlock(in_channels, key_channels, scale)
        self.conv_bn_dropout = nn.Sequential(
            ConvReLU(2 * in_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.Dropout2d(dropout),
        )

    def forward(self, feats, context):
        context = self.object_context_block(feats, context)
        output = self.conv_bn_dropout(torch.cat([context, feats], dim=1))
        return output


class SpatialGather(nn.Module):
    def __init__(self, n_classes: int = 9, scale: int = 1):
        super(SpatialGather, self).__init__()
        self.cls_num = n_classes
        self.scale = scale

    def forward(self, features, probs):
        n, k, _, _ = probs.shape
        _, c, _, _ = features.shape
        probs = probs.view(n, k, -1)
        features = features.view(n, -1, c)
        probs = torch.softmax(self.scale * probs, dim=-1)
        ocr_context = torch.matmul(probs, features)
        return ocr_context.view(n, k, c, 1)


class DANet_OCR(nn.Module):
    def __init__(self, n_classes: int = 9, channels: int = 256, ocr_mid_channels: int = 256, ocr_key_channels: int = 256):
        super(DANet_OCR, self).__init__()
        self.danet = DANet(channels, aux=False)
        self.soft_object_regions = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(channels, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.pixel_representations = ConvReLU(channels, out_channels=ocr_mid_channels, kernel_size=3)
        self.object_region_representations = SpatialGather(n_classes)
        self.object_contextual_representations = SpatialOCR(
            in_channels=ocr_mid_channels,
            key_channels=ocr_key_channels,
            out_channels=ocr_mid_channels,
            scale=1,
            dropout=0.5,
        )
        self.augmented_representation = nn.Conv2d(ocr_mid_channels, n_classes, kernel_size=1, padding=0)

    def forward(self, x):
        features = self.danet(x)[0]
        out_aux = self.soft_object_regions(features)
        features = self.pixel_representations(features)
        context = self.object_region_representations(features, out_aux)
        features = self.object_contextual_representations(features, context)
        out = self.augmented_representation(features)
        return out


if __name__ == '__main__':
    from torchsummary import summary
    # a = DANet_OCR(n_classes=9).cuda()
    # summary(a, (3, 512, 512))
    a = NonLocalPositionAttention().cuda()
    summary(a, (2048, 16, 16))
