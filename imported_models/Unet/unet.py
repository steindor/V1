
# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
                 batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2**(wf+i),
                                                padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2**(wf+i), up_mode,
                                            padding, batch_norm))
            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                                         stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_size, out_size, kernel_size=1))

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from IPython.core.debugger import set_trace


# class UnetConv2(nn.Module):
#     def __init__(self, in_size, out_size, is_batchnorm):
#         super(UnetConv2, self).__init__()

#         if is_batchnorm:
#             self.conv1 = nn.Sequential(
#                 # in, out, kernel, stride, padding
#                 nn.Conv2d(in_size, out_size, 3, 1, 1),
#                 nn.BatchNorm2d(out_size),
#                 nn.ReLU(),
#             )
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(out_size, out_size, 3, 1, 1),
#                 nn.BatchNorm2d(out_size),
#                 nn.ReLU(),
#             )
#         else:
#             self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0), nn.ReLU())
#             self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0), nn.ReLU()
#             )

#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         outputs = self.conv2(outputs)
#         return outputs


# class UnetUp(nn.Module):
#     def __init__(self, in_size, out_size, is_deconv):
#         super(UnetUp, self).__init__()
#         self.conv = UnetConv2(in_size, out_size, False)
#         if is_deconv:
#             self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
#         else:
#             self.up = nn.UpsamplingBilinear2d(scale_factor=2)

#     def forward(self, inputs1, inputs2):
#         outputs2 = self.up(inputs2)
#         offset = outputs2.size()[2] - inputs1.size()[2]
#         padding = 2 * [offset // 2, offset // 2]
        
#         outputs1 = F.pad(inputs1, padding)
#         return self.conv(torch.cat([outputs1, outputs2], 1))


# class Unet(nn.Module):

#     def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True):
#         super(Unet, self).__init__()

#         self.feature_scale = feature_scale
#         self.n_classes = n_classes
#         self.is_deconv = is_deconv
#         self.in_channels = in_channels
#         self.is_batchnorm = is_batchnorm

#         filters = [64,128,256,512,1024]
#         filters = [int(x / self.feature_scale) for x in filters]

#         #downsampling
#         self.conv1 = UnetConv2(self.in_channels, filters[0], self.is_batchnorm)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)

#         self.conv2 = UnetConv2(filters[0], filters[1], self.is_batchnorm)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)

#         self.conv3 = UnetConv2(filters[1], filters[2], self.is_batchnorm)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2)

#         self.conv4 = UnetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2)

#         self.center = UnetConv2(filters[3], filters[4], self.is_batchnorm)

#         self.up_concat4 = UnetUp(filters[4], filters[3], self.is_deconv)
#         self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
#         self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
#         self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

#         self.final = nn.Conv2d(filters[0], n_classes, 1)

#     def forward(self, inputs):
#         conv1 = self.conv1(inputs)
#         maxpool1 = self.maxpool1(conv1)

#         conv2 = self.conv2(maxpool1)
#         maxpool2 = self.maxpool2(conv2)

#         conv3 = self.conv3(maxpool2)
#         maxpool3 = self.maxpool3(conv3)

#         conv4 = self.conv4(maxpool3)
#         maxpool4 = self.maxpool4(conv4)

#         center = self.center(maxpool4)

#         up4 = self.up_concat4(conv4, center)
#         up3 = self.up_concat3(conv3, up4)
#         up2 = self.up_concat2(conv2, up3)
#         up1 = self.up_concat1(conv1, up2)

#         final = self.final(up1)

#         return final
