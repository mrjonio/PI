# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

class UNETsota(nn.Module):

    def __init__(self):

        super().__init__()
        #encoder
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        #middle
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(1024)

        #decoder
        self.deconv6_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.deconv7_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.deconv8_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.deconv9_1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv9_3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        #self.conv9_4 = nn.Conv2d(64, 1, kernel_size=1)
        self.outputer = nn.Tanh()

    def forward(self, x):

        #encoder
        #in 256x256
        y = self.bn1(self.relu(self.conv1_1(x)))
        y1 = self.bn1(self.relu(self.conv1_2(y)))
        y = self.pool(y1)
        #in 128x128
        y = self.bn1(self.relu(self.conv2_1(y)))
        y = self.bn2(self.relu(self.conv2_2(y)))
        y2 = self.bn2(self.relu(self.conv2_3(y)))
        y = self.pool(y2)
        #in 64x64
        y = self.bn2(self.relu(self.conv3_1(y)))
        y = self.bn3(self.relu(self.conv3_2(y)))
        y3 = self.bn3(self.relu(self.conv3_3(y)))
        y = self.pool(y3)
        #in 32x32
        y = self.bn3(self.relu(self.conv4_1(y)))
        y = self.bn4(self.relu(self.conv4_2(y)))
        y4 = self.bn4(self.relu(self.conv4_3(y)))
        y = self.pool(y4)
        #in 16x16

        x = torch.flatten(y, 1)
        #middle
        y = self.bn4(self.relu(self.conv5_1(y)))
        y = self.bn5(self.relu(self.conv5_2(y)))
        y = self.bn4(self.relu(self.conv5_3(y)))



        #decoder


        y = self.bn4(self.relu(self.deconv6_1(y)))
        y = y + y4
        y = self.bn4(self.relu(self.conv6_2(y)))
        y = self.bn4(self.relu(self.conv6_3(y)))
        y = self.bn3(self.relu(self.conv6_4(y)))


        y = self.bn3(self.relu(self.deconv7_1(y)))
        y = y + y3
        y = self.bn3(self.relu(self.conv7_2(y)))
        y = self.bn3(self.relu(self.conv7_3(y)))
        y = self.bn2(self.relu(self.conv7_4(y)))


        y = self.bn2(self.relu(self.deconv8_1(y)))
        y = y + y2
        y = self.bn2(self.relu(self.conv8_2(y)))
        y = self.bn2(self.relu(self.conv8_3(y)))
        y = self.bn1(self.relu(self.conv8_4(y)))


        y = self.bn1(self.relu(self.deconv9_1(y)))
        y = y + y1
        y = self.bn1(self.relu(self.conv9_2(y)))
        #y = self.bn1(self.relu(self.conv9_3(y)))
        y = self.outputer(self.relu(self.conv9_3(y)))

        return y, x

class UNETmodel(nn.Module):

    def __init__(self, pretrained_net):
        super().__init__()

        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)
        self.outputer = nn.Tanh()

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        score = self.outputer(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

