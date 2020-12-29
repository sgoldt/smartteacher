#!/usr/bin/env python3
#
# Models to train two-layer nets on (DCGAN, ResNet) generative model.
#
# Date: December 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import math

import torch
import torch.nn as nn

import torchvision.models.resnet


SQRT2 = 1.414213562
SQRT2OVERPI = 0.797884560802865


def erfscaled(x):
    """
    Re-scaled error function activation function.

    Useful to make the connection with an ODE description.
    """
    return torch.erf(x / SQRT2)


def dgdx_erfscaled(x):
    """
    Re-scaled error function activation function.

    Useful to make the connection with an ODE description.
    """
    return SQRT2OVERPI * torch.exp(-(x ** 2) / 2)


class TwoLayer(nn.Module):
    def __init__(self, g, N, K, std0w=1, std0v=1):
        """
        Parameters:
        -----------

        N : input dimension
        K : number of hidden nodes
        std0w :
           std dev of the initial FIRST-layer (Gaussian) weights.
        std0v :
           std dev of the initial SECOND-layer (Gaussian) weights.
        """
        super(TwoLayer, self).__init__()
        self.N = N
        self.K = K
        self.g = g

        self.fc1 = nn.Linear(N, K, bias=False)
        self.fc2 = nn.Linear(K, 1, bias=False)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=std0w)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=std0v)

    def forward(self, x):
        # input to hidden
        x = self.g(self.fc1(x) / math.sqrt(self.N))
        x = self.fc2(x)
        return x

    def nu_y(self, x):
        """
        Computes the pre-activation of the teacher.
        """
        nu = self.fc1(x) / math.sqrt(self.N)
        y = self.g(nu)
        y = self.fc2(y)
        return nu, y

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class ScalarResnet(torchvision.models.resnet.ResNet):
    def __init__(self, num_classes, pretrained=False, progress=True, **kwargs):
        super(ScalarResnet, self).__init__(
            torchvision.models.resnet.BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            **kwargs
        )
        self.input_dim = (1, 32, 32)
        self.num_classes = num_classes
        self.N = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        self.D = max(self.fc.weight.data.shape)

        # Two tricks to get better accuracy, taken from Joost van Amersfoort
        # https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
        self.conv1 = torch.nn.Conv2d(
            self.input_dim[0], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = torch.nn.Identity()

        # add a batch-norm layer before the last fully connected layer
        self.bnz = nn.BatchNorm1d(self.D, affine=False, track_running_stats=False)
        # turn off the bias in the final fully-connected layer
        self.fc.bias = None

        nn.init.normal_(self.fc.weight.data, 0, 1)  # ensure the correct scaling here!

    def preprocess(self, x, block=4):
        """
        Propagates the given inputs x through the ResNet until we hit the fully-connected
        layer that projects things down to the K classes.  Code is taken directly from
        _forward_impl of pyTorch ResNet implementation, only x = self.fc(x) is missing
        at the end.

        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if block > 0:
            x = self.layer1(x)
        if block > 1:
            x = self.layer2(x)
        if block > 2:
            x = self.layer3(x)
        if block > 3:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.bnz(x)  # an additional BatchNorm layer to ensure that z has zero mean

        return x

    def forward(self, x):
        x = self.nu(x)
        x = erfscaled(x)
        return x

    def nu(self, x):
        """
        Computes the pre-activation of the teacher.
        """
        x = self.preprocess(x)
        x = self.fc(x / math.sqrt(self.D))
        return x

    def nu_y(self, x):
        """
        Computes the pre-activation of the teacher.
        """
        nu = self.preprocess(x)
        nu = self.fc(nu / math.sqrt(self.D))
        y = erfscaled(nu)
        return nu, y

    def freeze(self):
        """
        Deactivates automatic differentiation for all parameters.

        Useful when defining a teacher.
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Activates automatic differentiation for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = True
