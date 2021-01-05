#!/usr/bin/env python3
"""
Various tests for the models used in this project.

Author: Sebastian Goldt <goldt.sebastian@gmail.com>

December 2020
"""

import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ScalarResnet, erfscaled


def explicit_forward(student, Ds, inputs):
    """
    Performs an explicit forward pass for a fully-connected network with at least
    two layers of weights.
    """
    with torch.no_grad():
        # forward pass of the student
        L = student.num_layers
        a = [None] * L  # activations for each layer
        h = [None] * L  # activities for each layer

        if L > 2:
            w0 = student.layers[0].weight.data
            a[0] = w0 @ inputs.T / math.sqrt(Ds[0])
            h[0] = student.g(a[0])
        for d in range(1, L - 2):
            w = student.layers[d].weight.data
            a[d] = w @ h[d - 1] / math.sqrt(Ds[d])
            h[d] = student.g(a[d])
        w = student.fc.weight.data
        a[L - 2] = w @ (h[L - 3] if L > 2 else inputs.T) / math.sqrt(Ds[-1])
        h[L - 2] = student.g(a[L - 2])
        v = student.v.weight.data
        a[L - 1] = v @ h[L - 2]
        h[L - 1] = a[L - 1]
        ys = h[L - 1].t()

    return ys


class ScalarResnetTests(unittest.TestCase):
    def test_outputs_using_preprocess(self):
        bs = 4
        num_classes = 1

        student = ScalarResnet(erfscaled, num_classes)

        with torch.no_grad():
            xs = torch.randn((bs,) + student.input_dim)
            ys = student(xs)

            w = student.fc.weight.data
            v = student.v.weight.data

            zs = student.preprocess(xs)
            lambdas = w @ zs.T / math.sqrt(student.D)
            glambdas = erfscaled(lambdas)
            explicit_ys = v @ glambdas

        diff = torch.sum((ys - explicit_ys.T) ** 2) / torch.sum(ys ** 2)
        self.assertTrue(diff.item() < 1e-6)

    def test_preprocess_block0(self):
        bs = 128
        num_classes = 1

        student = ScalarResnet(erfscaled, num_classes)

        with torch.no_grad():
            xs = torch.randn((bs,) + student.input_dim)
            zs = student.preprocess(xs, 0)

            # I know, I know, this is testing internal logic...
            xs = student._resnet.conv1(xs)
            xs = student._resnet.bn1(xs)
            xs = student._resnet.relu(xs)
            xs = student._resnet.maxpool(xs)
            # x = self.layer1(x)
            # x = self.layer2(x)
            # x = self.layer3(x)
            # x = self.layer4(x)

            xs = student._resnet.avgpool(xs)
            xs = torch.flatten(xs, 1)

            xs = student.bnz(
                xs
            )  # an additional BatchNorm layer to ensure that z has zero mean

        self.assertTrue(torch.mean(xs) < 1e-2)
        self.assertTrue(abs(torch.std(xs) - 1) < 1e-1)
        self.assertTrue(torch.allclose(xs, zs))

    def test_preprocess_block1(self):
        bs = 128
        num_classes = 3

        student = ScalarResnet(erfscaled, num_classes)

        with torch.no_grad():
            xs = torch.randn((bs,) + student.input_dim)
            zs = student.preprocess(xs, 1)

            # I know, I know, this is testing internal logic...
            xs = student._resnet.conv1(xs)
            xs = student._resnet.bn1(xs)
            xs = student._resnet.relu(xs)
            xs = student._resnet.maxpool(xs)
            xs = student._resnet.layer1(xs)
            # x = self.layer2(x)
            # x = self.layer3(x)
            # x = self.layer4(x)

            xs = student._resnet.avgpool(xs)
            xs = torch.flatten(xs, 1)

            xs = student.bnz(
                xs
            )  # an additional BatchNorm layer to ensure that z has zero mean

        self.assertTrue(torch.mean(xs) < 1e-2)
        self.assertTrue(abs(torch.std(xs) - 1) < 1e-1)
        self.assertTrue(torch.allclose(xs, zs))

    def test_preprocess_block3(self):
        bs = 128
        num_classes = 3

        student = ScalarResnet(erfscaled, num_classes)

        with torch.no_grad():
            xs = torch.randn((bs,) + student.input_dim)
            zs = student.preprocess(xs, 3)

            # I know, I know, this is testing internal logic...
            xs = student._resnet.conv1(xs)
            xs = student._resnet.bn1(xs)
            xs = student._resnet.relu(xs)
            xs = student._resnet.maxpool(xs)
            xs = student._resnet.layer1(xs)
            xs = student._resnet.layer2(xs)
            xs = student._resnet.layer3(xs)
            # x = self.layer4(x)

            xs = student._resnet.avgpool(xs)
            xs = torch.flatten(xs, 1)

            xs = student.bnz(
                xs
            )  # an additional BatchNorm layer to ensure that z has zero mean

        self.assertTrue(torch.mean(xs) < 1e-2)
        self.assertTrue(abs(torch.std(xs) - 1) < 1e-1)
        self.assertTrue(torch.allclose(xs, zs))


if __name__ == "__main__":
    unittest.main()
