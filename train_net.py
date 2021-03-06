import os
import json
import sys
import math
import argparse

import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from tqdm import tqdm

from my_dataset import MyDataSet
from model import shufflenet_v2_x1
from data_preparation import split_data


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item()


def train(args):
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 训练集数据地址
    image_path = args.data_path
    assert os.path.exists(image_path), "{} 路径不存在.".format(image_path)

    train_images_path, train_images_label, val_images_path, val_images_label = split_data(args.data_path)

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_label=train_images_label,
                              transform=data_transform['train'])
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_label=val_images_label,
                            transform=data_transform['val'])

    batch_size = args.batch_size

    # os.cpu_count()Python中的方法用于获取系统中的CPU数量
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=nw,
                              collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=nw,
                            collate_fn=val_dataset.collate_fn)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    isTrain = True
    if isTrain:
        net = shufflenet_v2_x1(num_classes=args.num_classes).to(device)

        # 加载预训练权重
        if os.path.exists(args.weights):
            pre_weights = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in pre_weights.items()
                                 if pre_weights.state_dict()[k].numel() == v.numel()}
            print(net.load_state_dict(load_weights_dict, strict=False))

        if args.freeze_layers:
            for name, param in net.named_parameters():
                if 'fc' not in name:
                    param.requires_grad_(False)

        pg = [p for p in net.parameters() if p.requires_grad]

        optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.0001)

        save_path = "./model_data.pth"

        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        for epoch in range(args.epochs):
            mean_loss = train_one_epoch(model=net, optimizer=optimizer, data_loader=train_loader, device=device,
                                        epoch=epoch)
            scheduler.step()

            # validate
            sum_num = evaluate(model=net,
                               data_loader=val_loader,
                               device=device)

            acc = sum_num / len(val_dataset)
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))

            torch.save(net.state_dict(), save_path)

        print('训练完成')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lrf', type=float, default=0.1)

    parser.add_argument('--data-path', type=str, default="/home/lee/pyCode/dl_data/flower_photos")
    parser.add_argument('--weights', type=str, default='../model_weight/shufflenetv2_x1.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    train(args)
