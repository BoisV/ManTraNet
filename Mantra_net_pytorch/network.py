from collections import OrderedDict
import torch
from torch import nn


class CombinedConv2D(nn.Module):
    def __init__(self):
        kernel = []
        srm_list = self.__get_stm_list()
        for idx, srm in enumerate(srm_list):
            for ch in range(3):
                this_ch_kernel = torch.zeros([5, 5, 3], dtype=torch.float32)
                this_ch_kernel[:, :, ch] = srm
                kernel.append(this_ch_kernel)
        kernel = torch.stack(kernel, dim=-1)
        srm_kernel = nn.Parameter(
            kernel, dtype=torch.float32, requires_grad=False)

    def __get_stm_list(self, dtype=torch.float32):
        srm1 = torch.zeros([5, 5], dtype=dtype)
        srm1[1:-1, 1:-1] = torch.tensor([[-1, 2, -1],
                                         [2, -4, 2],
                                         [-1, 2, -1]])
        srm1 /= 4

        srm2 = torch.tensor([[-1, 2, -2, 2, -1],
                             [2, -6, 8, -6, 2],
                             [-2, 8, -12, 8, -2],
                             [2, -6, 8, -6, 2],
                             [-1, 2, -2, 2, -1]], dtype=dtype)
        srm2 /= 12

        srm3 = torch.zeros([5, 5], dtype=dtype)
        srm3[2, 1:-1] = torch.tensor([1, -2, 1])
        srm3 /= 2
        return [srm1, srm2, srm3]


class ZPool2D(nn.Module):
    def __init__(self, kernel_size, epsilon):
        super(ZPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, X):
        mu = self.avgpool(X)
        sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)).sum() / (X.shape[-2] * X.shape[-1]))
        D = X - mu
        return D / sigma


class ZPool2DGlobal(nn.Module):
    def __init__(self):
        super(ZPool2D, self).__init__()

    def forward(self, X):
        mu = torch.mean(X, dim=(-2, -1))
        D = X - mu
        sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)).sum() / (X.shape[-2] * X.shape[-1]))
        return D / sigma

class MantraNet(nn.Module):
    def __init__(self, Featex, pool_size_list=[7, 15, 31], apply_normalization=True):
        self.rf = Featex
        self.outlierTrans = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=(1, 1), bias=False)
        self.bnorm = nn.BatchNorm2d(num_features=64)
        self.zpoolglobal = ZPool2DGlobal()
        self.zpools = nn.Sequential(OrderedDict([
            ('ZPool2D@{}x{}'.format(i), ZPool2D(i)) for i in pool_size_list
        ]))

if __name__ == "__main__":
    X = torch.tensor([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]]], dtype=torch.float32)
    print(X.shape)

    zpool2d = ZPool2D(1, 1, 3)
    y = zpool2d(X)
    print(y)
