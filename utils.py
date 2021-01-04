# Various utility functions for smart teachers.
#
# Date: January 2021
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import math
import os

import torch
from torchvision import datasets, transforms

from models import ScalarResnet, TwoLayer, ConvNet, MLP

teachers = "twolayer | mlp | convnet | resnet18"
generators = "iid | dcgan_rand | dcgan_cifar100_grey"

class IdentityTransform(torch.nn.Module):
    """Dummy image transform that doesn't do anything.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading
    dimensions
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image not to be converted.

        Returns:
            PIL Image or Tensor: The unchanged image
        """
        return img

    def __repr__(self):
        return self.__class__.__name__


def binarise(x):
    x = 2 * (x % 2) - 1
    return x

def I2_erf(c00, c01, c11):
    return 2.0 / math.pi * torch.asin(c01 / (math.sqrt(1 + c00) * math.sqrt(1 + c11)))


def erfscaled(x):
    return torch.erf(x / math.sqrt(2))


def get_eg_analytical(Q, R, T, A, v):
    """
    Computes the analytical expression for the generalisation error of erf teacher
    and student with the given order parameters.

    Parameters:
    -----------
    Q: student-student overlap
    R: teacher-student overlap
    T: teacher-teacher overlap
    A: teacher second layer weights
    v: student second layer weights
    """
    eg_analytical = 0
    # student-student overlaps
    sqrtQ = torch.sqrt(1 + Q.diag())
    norm = torch.ger(sqrtQ, sqrtQ)
    eg_analytical += torch.sum((v.t() @ v) * torch.asin(Q / norm))
    # teacher-teacher overlaps
    sqrtT = torch.sqrt(1 + T.diag())
    norm = torch.ger(sqrtT, sqrtT)
    eg_analytical += torch.sum((A.t() @ A) * torch.asin(T / norm))
    # student-teacher overlaps
    norm = torch.ger(sqrtQ, sqrtT)
    eg_analytical -= 2.0 * torch.sum((v.t() @ A) * torch.asin(R / norm))
    return eg_analytical / math.pi


def get_teacher(name, N, M, pretrained_on=None, device="cpu"):
    """
    Load a teacher with the given name. If pretrained is not None, will try to preload
    the dataset name given.

    name : twolayer | mlp | convnet | resnet18
    pretrained_on : rand | cifar100
    """
    teacher = None
    if name == "twolayer":
        teacher = TwoLayer(N, M, std0w=1, std0v=1)
    elif name == "mlp":
        teacher = MLP(N, M)
    elif name == "convnet":
        teacher = ConvNet(erfscaled, M)
    elif name == "resnet18":
        num_classes = 1
        teacher = ScalarResnet(num_classes)
    else:
        raise ValueError("Did not recognise the teacher description.")

    if pretrained_on is not None:
        teacher.load_state_dict(
            torch.load("./models/%s_%s.pt" % (name, pretrained_on), map_location=device)
        )

    return teacher


def get_dataset(root, name="cifar10", grayscale=False):
    """
    We use the values of the transform from the training of the dcGAN.
    """
    target_transform = binarise

    num_channels = 1 if grayscale else 3

    train_transform = transforms.Compose(
        [
            transforms.Grayscale() if grayscale else IdentityTransform(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * num_channels, (0.5,) * num_channels),
        ]
    )

    try:
        dataset_object = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    except KeyError as key_error:
        raise ValueError("Don't know about dataset %s." % name) from key_error

    train_dataset = dataset_object(
        root + os.sep + name,
        train=True,
        target_transform=target_transform,
        transform=train_transform,
        download=True,
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale() if grayscale else IdentityTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * num_channels, (0.5,) * num_channels),
        ]
    )

    test_dataset = dataset_object(
        root + os.sep + name,
        train=False,
        target_transform=target_transform,
        transform=test_transform,
        download=True,
    )

    return train_dataset, test_dataset