#!/usr/bin/env python3
#
# Training a scalar ResNet 18 on odd-even CIFAR10, based on a script by
# Joost van Amersfoort (https://gist.github.com/y0ast)
# https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
#
# Date: December 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import ScalarResnet


def binarise(x):
    x = 2 * (x % 2) - 1
    return x


def get_CIFAR10(root):
    """
    The transforms and corresponding parameters are taken from a script by
    Joost van Amersfoort (https://gist.github.com/y0ast)
    https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
    """

    target_transform = binarise

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root,
        train=True,
        target_transform=target_transform,
        transform=train_transform,
        download=True,
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root,
        train=False,
        target_transform=target_transform,
        transform=test_transform,
        download=True,
    )

    return train_dataset, test_dataset


def train(model, train_loader, optimizer, epoch, logfile, device):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.to(device)
        target = target.to(device).unsqueeze(-1)

        optimizer.zero_grad()

        prediction = model(data)
        loss = F.mse_loss(prediction, target.float())

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    log(f"Epoch: {epoch}:", logfile)
    log(f"Train Set: Average Loss: {avg_loss:.2f}", logfile)


def test(model, test_loader, logfile, device):
    model.eval()

    loss = 0
    correct = 0

    for data, target in tqdm(test_loader):
        with torch.no_grad():
            data = data.to(device)
            target = target.to(device).unsqueeze(-1)

            prediction = model(data)
            loss += F.mse_loss(prediction, target.float(), reduction="sum")

            prediction = torch.sign(prediction)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    log(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        ),
        logfile,
    )

    return loss, percentage_correct


def log(msg, logfile):
    """
    Print log message to  stdout and the given logfile.
    """
    print(msg)
    logfile.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    device_help = "which device to run on: 'cuda' or 'cpu'"
    parser.add_argument("--device", "-d", help=device_help)
    parser.add_argument(
        "--dataroot", default="~/datasets/cifar10", help="path to CIFAR10"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="learning rate (default: 0.05)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)

    train_dataset, test_dataset = get_CIFAR10(args.dataroot)

    kwargs = {"num_workers": 2, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2000, shuffle=False, **kwargs
    )

    num_classes = 1  # one output head for odd-even classification
    model = ScalarResnet(1)
    model = model.to(device)

    # output file + welcome message
    fname_root = "train_scalarresnet18_cifar10_s%d" % (args.seed)
    logfile = open(fname_root + ".log", "w", buffering=1)
    welcome = "# Training a scalar Resnet18 on odd-even discrimination for CIFAR10\n"
    welcome += "# Using device:" + str(device) + "; seed=" + str(args.seed)
    log(welcome, logfile)

    milestones = [25, 40]

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    for epoch in range(1, args.epochs + 1):
        test(model, test_loader, logfile, device)
        train(model, train_loader, optimizer, epoch, logfile, device)

        scheduler.step()

        torch.save(model.state_dict(), "models/scalarresnet18_cifar10_ep%d.pt" % epoch)


if __name__ == "__main__":
    main()
