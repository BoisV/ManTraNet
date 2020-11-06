from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from convlstm import ConvLSTM


class SRMConv2D(nn.Module):
    def __init__(self) -> None:
        super(SRMConv2D, self).__init__()

    def forward(self):
        pass


class BayarConv2D(nn.Module):
    def __init__(self) -> None:
        super(BayarConv2D, self).__init__()

    def forward(self):
        pass


class CombinedConv2D(nn.Module):
    def __init__(self):
        super(CombinedConv2D, self).__init__()

    def forward(self):
        pass


class FeatexVGG16(nn.Module):
    def __init__(self, type=1):
        super(FeatexVGG16).__init__()
        # block1
        self.block1 = nn.Sequential(OrderedDict([
            ('b1c1', CombinedConv2D()),
            ('b1c2', nn.Conv2d(
                in_channels=32, out_channels=32, stride=1, padding=1)),
            ('b1ac', nn.ReLU)
        ]))

        # block2
        self.block2 = nn.Sequential(OrderedDict([
            ('b2c1', nn.Conv2d(
                in_channels=32, out_channels=64, stride=1, padding=1)),
            ('b2ac1', nn.ReLU()),
            ('b2c2', nn.Conv2d(in_channels=64,
                               out_channels=64, stride=1, padding=1)),
            ('b2ac2', nn.ReLU)
        ]))

        # block3
        self.block3 = nn.Sequential(OrderedDict([
            ('b3c1', nn.Conv2d(
            in_channels=64, out_channels=128, stride=1, padding=1)),
            ('b3ac1', nn.ReLU()),
            ('b3c2', nn.Conv2d(in_channels=128, out_channels=128, stride=1, padding=1)),
            ('b3ac2', nn.ReLU()),
            ('b3c3', nn.Conv2d(in_channels=128,
                               out_channels=128, stride=1, padding=1)),
            ('b3ac3', nn.ReLU())
        ]))


        # block4
        self.block4 = nn.Sequential(OrderedDict([
                ('b4c1', nn.Conv2d(
                    in_channels=128, out_channels=256, stride=1, padding=1)),
                ('b4ac1', nn.ReLU()),
                ('b4c2', nn.Conv2d(in_channels=256,
                                   out_channels=256, stride=1, padding=1)),
                ('b4ac2', nn.ReLU()),
                ('b4c3', nn.Conv2d(in_channels=256,
                                   out_channels=256, stride=1, padding=1)),
                ('b4ac3', nn.ReLU())
        ]))

        # block5
        self.block5 = nn.Sequence(OrderedDict([
            ('b5c1', nn.Conv2d(
                in_channels=256, out_channels=256, stride=1, padding=1))
            ('b5ac1', nn.ReLU()),
            ('b5c2', nn.Conv2d(in_channels=256,
                               out_channels=256, stride=1, padding=1)),
            ('b5ac2', nn.ReLU())
        ]))

        self.transform = nn.Conv2d(in_channels=256, out_channels=256, stride=1, padding=1)
        self.activation = None if type>=1 else nn.Tanh()


    def farward(self, X):
        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)
        X = self.block4(X)
        X = self.block5(X)
        X = self.transform(X)
        if self.activation is not None:
            X = self.activation(X)
        return nn.functional.normalize(X, 2, dim=-1)



class ZPool2D(nn.Module):
    def __init__(self, kernel_size):
        super(ZPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, X):
        mu = self.avgpool(X)
        sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)
                            ).sum() / (X.shape[-2] * X.shape[-1]))
        D = X - mu
        return D / sigma


class ZPool2DGlobal(nn.Module):
    def __init__(self):
        super(ZPool2DGlobal, self).__init__()

    def forward(self, X):
        mu = torch.mean(X, dim=(2, 3), keepdim=True)
        D = X - mu
        sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)
                            ).sum() / (X.shape[-2] * X.shape[-1]))
        return D / sigma


class MantraNet(nn.Module):
    def __init__(self, Featex=None, pool_size_list=[7, 15, 31]):
        super(MantraNet, self).__init__()
        # self.rf = Featex
        self.outlierTrans = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(1, 1), bias=False)
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.zpoolglobal = ZPool2DGlobal()
        zpools = OrderedDict()
        for i in pool_size_list:
            name = 'ZPool2D@{}x{}'.format(i, i)
            zpools[name] = ZPool2D(i)
        self.zpools = nn.Sequential(zpools)
        self.cLSTM = ConvLSTM(64, 8, (7, 7), 1, batch_first=True)
        self.pred = nn.Conv2d(in_channels=8, out_channels=1,
                              kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        # X = self.rf(X)
        X = self.bnorm(self.outlierTrans(X))
        Z = []
        Z.append(torch.unsqueeze(self.zpoolglobal(X), dim=1))
        for index in range(len(self.zpools)-1, -1, -1):
            Z.append(torch.unsqueeze(self.zpools[index](X), dim=1))
        Z = torch.cat([i for i in Z], dim=1)
        last_output_list, _ = self.cLSTM(Z)
        X = last_output_list[0][:, -1, :, :, :]
        output = self.sigmoid(self.pred(X))
        return output


if __name__ == "__main__":
    X = torch.randn([1, 3, 256, 256])
    # zpool2d = ZPool2DGlobal()
    # zpool2d1 = ZPool2D(kernel_size=7)
    # zpool2d2 = ZPool2D(kernel_size=15)
    # zpool2d3 = ZPool2D(kernel_size=31)
    # y = zpool2d(X)
    # y1 = zpool2d1(X)
    # y2 = zpool2d2(X)
    # y3 = zpool2d3(X)
    # print(y.shape, y1.shape, y2.shape, y3.shape)
    # y = torch.unsqueeze(y, dim=1)
    # y3 = torch.unsqueeze(y3, dim=1)
    # y2 = torch.unsqueeze(y2, dim=1)
    # y1 = torch.unsqueeze(y1, dim=1)
    # Y = torch.cat([y, y3, y2, y1], dim=1)
    # hidden_state = torch.zeros_like(y)
    # conv2dlstm = ConvLSTM(3, 8, (7, 7), 1, batch_first=True)
    # last_output, last_state = conv2dlstm(Y)
    # output = last_output[0][:, -1, :, :, :]
    # print(output.shape)
    net = MantraNet()
    output = net(X)
    print(output.shape)
