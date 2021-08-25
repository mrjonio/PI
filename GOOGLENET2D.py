import torch
import torch.nn as nn
import torch.nn.functional as F

class GOOGLENET2Dmodel(nn.Module):


    def __init__(self):
        super().__init__()

#DEscomentar quando for fase 1
        #self.conv1          = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding= 1)
        self.conv1          = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding= 1)
        self.relu           = nn.ReLU(inplace=True)
        self.conv2          = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding = 1)
        self.bn1            = nn.BatchNorm2d(64)
        self.inception1     = InceptionB(64)
        self.incpetion2     = InceptionB(192)
        self.conv3          = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.conv4          = nn.Conv2d(64, 1, kernel_size=1)
        self.outputer       = nn.Tanh()


    def forward(self,x):

        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        y = self.relu(self.bn1(y))
        y = self.inception1(y)
        y = self.incpetion2(y)
        y = self.incpetion2(y)
        y = self.incpetion2(y)
        y = self.incpetion2(y)
        y = self.incpetion2(y)
        y = self.incpetion2(y)
        y = self.incpetion2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.outputer(y)

        return y

class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x2dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x2dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x2dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(224)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x2dbl = self.branch3x2dbl_1(x)
        branch3x2dbl = self.branch3x2dbl_2(branch3x2dbl)
        branch3x2dbl = self.branch3x2dbl_3(branch3x2dbl)

        concat = [branch1x1, branch5x5, branch3x2dbl]
        concat_out = torch.cat(concat, 1)

        return self.relu(self.bn(concat_out))

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5 = BasicConv2d(in_channels, 64, kernel_size=5, padding = 2)

        self.branch3x3 = BasicConv2d(in_channels, 64, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(192)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3 = self.branch3x3(x)

        concat = [branch1x1, branch5x5, branch3x3]
        concat_out = torch.cat(concat, 1)

        return self.relu(self.bn(concat_out))

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
