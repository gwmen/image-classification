import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import os
import argparse

import os
import sys
import torch
from torch.backends import cudnn

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # with hf-mirror
from modeling.backbones.models import create_model

_py_hash_seed_env = "PYTHONHASHSEED"


def seed_everything(seed=1234):
    """确保实验可重复性"""
    random.seed(seed)
    os.environ[_py_hash_seed_env] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")


class Baseline(nn.Module):
    def __init__(self, num_classes=10):
        super(Baseline, self).__init__()
        self.base, self.in_planes = create_model('resnet18')
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes)

    def forward(self, x):
        global_feat = self.base(x)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        return self.classifier(global_feat)


def train(epoch, net, optimizer, device, trainloader, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = train_loss / len(trainloader)

    return avg_loss, accuracy


def test(epoch, net, device, testloader, criterion, save_dir='./checkpoint'):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    avg_loss = test_loss / len(testloader)

    return avg_loss, accuracy


def save_checkpoint(state, filename):
    """保存检查点"""
    torch.save(state, filename)
    # print(f"检查点已保存: {filename}")


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='学习率')
    parser.add_argument('--resume', '-r', default=True, help='从检查点恢复')
    parser.add_argument('--resume_path', default='./checkpoint/ckpt_00005.pth',
                        type=str, help='检查点路径')
    parser.add_argument('--epochs', default=200, type=int, help='总训练轮数')
    parser.add_argument('--seed', default=1234, type=int, help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    seed_everything(args.seed)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # 模型
    model = Baseline()

    # 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练状态
    start_epoch = 0
    best_acc = 0.0
    checkpoint_dir = './checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 恢复训练
    if args.resume and os.path.exists(args.resume_path):
        print(f"==> 从检查点恢复: {args.resume_path}")

        # 加载检查点
        checkpoint = torch.load(args.resume_path, map_location=device)

        # 加载模型
        model.load_state_dict(checkpoint['net'])
        model = model.to(device)
        print(f"✓ 模型已加载")

        # 加载优化器
        optimizer.load_state_dict(checkpoint['opt'])

        # 修复优化器设备
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.device != device:
                    param.data = param.data.to(device)

        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        print(f"✓ 优化器已加载")

        # 加载调度器
        scheduler.load_state_dict(checkpoint['sche'])

        # 关键：设置调度器的last_epoch
        scheduler.last_epoch = checkpoint['epoch']

        # 恢复训练状态
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['acc']

        print(f"✓ 调度器已加载")
        print(f"  恢复的epoch: {checkpoint['epoch']}")
        print(f"  开始的epoch: {start_epoch}")
        print(f"  最佳准确率: {best_acc:.4f}")
        print(f"  当前学习率: {scheduler.get_last_lr()[0]:.6f}")
    else:
        # 从头开始
        model = model.to(device)
        print("==> 从头开始训练")
    import time
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        t1 = time.time()
        # 训练
        train_loss, train_acc = train(epoch, model, optimizer, device, trainloader, criterion)
        t2 = time.time()
        print(t2 - t1)
        # 测试
        test_loss, test_acc = test(epoch, model, device, testloader, criterion)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 保存最佳模型
        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     save_checkpoint({
        #         'net': model.state_dict(),
        #         'opt': optimizer.state_dict(),
        #         'sche': scheduler.state_dict(),
        #         'acc': test_acc,
        #         'epoch': epoch,
        #         'loss': test_loss,
        #     }, f'{checkpoint_dir}/best.pth')

        # 定期保存
        # if (epoch + 1) % 10 == 0:
        if 1:
            save_checkpoint({
                'net': model.state_dict(),
                'opt': optimizer.state_dict(),
                'sche': scheduler.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'loss': test_loss,
            }, f'{checkpoint_dir}/ckpt_{str(epoch + 1).zfill(5)}.pth')

        # 打印进度
        # print(f"Epoch: {epoch + 1:03d}/{args.epochs} | "
        #       f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        #       f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.2f}% | "
        #       f"LR: {current_lr:.6f}")

    print(f"\n训练完成! 最佳准确率: {best_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
# Epoch: 001/200 | Train Loss: 1.3955 | Train Acc: 50.28% | Test Loss: 1.2840 | Test Acc: 57.19% | LR: 0.099994
# Epoch: 002/200 | Train Loss: 0.9522 | Train Acc: 66.75% | Test Loss: 0.9226 | Test Acc: 67.76% | LR: 0.099975
# Epoch: 003/200 | Train Loss: 0.8085 | Train Acc: 72.08% | Test Loss: 0.7647 | Test Acc: 73.38% | LR: 0.099944
# Epoch: 004/200 | Train Loss: 0.7470 | Train Acc: 74.11% | Test Loss: 0.9129 | Test Acc: 68.50% | LR: 0.099901
# Epoch: 005/200 | Train Loss: 0.7255 | Train Acc: 74.88% | Test Loss: 0.7201 | Test Acc: 74.89% | LR: 0.099846
# Epoch: 006/200 | Train Loss: 0.7021 | Train Acc: 75.85% | Test Loss: 0.7710 | Test Acc: 73.24% | LR: 0.099778
# Epoch: 007/200 | Train Loss: 0.6989 | Train Acc: 75.96% | Test Loss: 0.7930 | Test Acc: 73.39% | LR: 0.099698
