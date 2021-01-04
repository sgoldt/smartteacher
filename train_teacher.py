#!/usr/bin/env python3
#
# Training various teachers on odd-even CIFAR100, based on a script by
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

import utils


def train(model, train_loader, optimizer, epoch, logfile, device):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.to(device)
        target = target.to(device).unsqueeze(-1)

        if not model.requires_2d_input:
            data = data.reshape(-1, model.input_dim)

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

            if not model.requires_2d_input:
                data = data.reshape(-1, model.input_dim)

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
    parser.add_argument("--teacher", default="twolayer", help=utils.teachers)
    parser.add_argument("--dataroot", default="~/datasets", help="path to datasets")
    parser.add_argument("--dataset", default="cifar10", help="cifar10 | cifar100")
    parser.add_argument(
        "--grayscale", help="transform images to grayscale", action="store_true"
    )
    parser.add_argument(
        "-M", type=int, default=1, help="teacher hidden nodes M (default=1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.05, help="learning rate (default: 0.05)"
    )
    parser.add_argument("--device", "-d", help=device_help)
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_dataset, test_dataset = utils.get_dataset(args.dataroot, name=args.dataset, grayscale=args.grayscale)

    kwargs = {"num_workers": 2, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2000, shuffle=False, **kwargs
    )

    num_channels = 1 if args.grayscale else 3
    N = 32 * 32 * num_channels
    M = {"twolayer": args.M, "mlp": args.M, "convnet": args.M, "resnet18": 1}[args.teacher]
    kwargs = {"input_dim": [num_channels, 32, 32]}
    model = utils.get_model(args.teacher, N, M, **kwargs)
    model = model.to(device)

    # output file + welcome message
    dataset_desc = args.dataset + ("_gray" if args.grayscale else "")
    fname_root = "train_%s_%s_s%d" % (args.teacher, dataset_desc, args.seed)
    logfile = open(fname_root + ".log", "w", buffering=1)
    welcome = "# Training a %s on odd-even discrimination on %s\n" % (
        args.teacher,
        dataset_desc,
    )
    welcome += "# Using device:" + str(device) + "; seed=" + str(args.seed)
    welcome += "\n# " + str(model).replace("\n", "\n# ")
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

        torch.save(
            model.state_dict(),
            "models/%s_%s_ep%d.pt" % (args.teacher, dataset_desc, epoch),
        )


if __name__ == "__main__":
    main()
