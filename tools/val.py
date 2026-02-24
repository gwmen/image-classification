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
from modeling.backbones.model_generator import fetch_model


class TrainingStateSaver:
    """训练状态保存器"""

    @staticmethod
    def save_full_state(epoch, model, optimizer, scheduler, trainloader,
                        testloader, criterion, best_acc, path):
        """保存完整的训练状态"""
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_acc': best_acc,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'cuda_random_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'cuda_random_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # 保存数据加载器状态
        if hasattr(trainloader, 'sampler'):
            if hasattr(trainloader.sampler, 'state_dict'):
                state['train_sampler_state'] = trainloader.sampler.state_dict()

        torch.save(state, path)
        print(f"完整训练状态已保存: {path}")
        return state

    @staticmethod
    def load_full_state(path, model, optimizer, scheduler, trainloader, device):
        """加载完整的训练状态"""
        if not os.path.exists(path):
            return 0, 0.0

        state = torch.load(path, map_location='cpu')

        # 恢复随机状态
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_random_state'])
        torch.set_rng_state(state['torch_random_state'])

        if torch.cuda.is_available():
            if state['cuda_random_state'] is not None:
                torch.cuda.set_rng_state(state['cuda_random_state'])
            if state['cuda_random_state_all'] is not None:
                torch.cuda.set_rng_state_all(state['cuda_random_state_all'])

        # 恢复模型
        model.load_state_dict(state['model_state'])
        model = model.to(device)

        # 恢复优化器
        optimizer.load_state_dict(state['optimizer_state'])

        # 确保优化器在正确设备
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.device != device:
                    param.data = param.data.to(device)

        for opt_state in optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(device)

        # 恢复调度器
        scheduler.load_state_dict(state['scheduler_state'])
        scheduler.last_epoch = state['epoch']

        # 恢复数据加载器状态
        if 'train_sampler_state' in state and hasattr(trainloader, 'sampler'):
            if hasattr(trainloader.sampler, 'load_state_dict'):
                trainloader.sampler.load_state_dict(state['train_sampler_state'])

        print(f"完整训练状态已恢复")
        print(f"  Epoch: {state['epoch']}")
        print(f"  最佳准确率: {state['best_acc']:.4f}")
        print(f"  当前学习率: {scheduler.get_last_lr()[0]:.6f}")

        return state['epoch'], state['best_acc']


def setup_deterministic_training(seed=1234):
    """设置确定性训练"""
    # Python随机
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Numpy随机
    np.random.seed(seed)

    # PyTorch随机
    torch.manual_seed(seed)

    # CUDA随机
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 设置确定性算法
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 设置卷积算法
        torch.backends.cudnn.enabled = True

    print(f"确定性训练设置完成，种子: {seed}")


class DeterministicDataLoader:
    """确定性的数据加载器"""

    @staticmethod
    def create(train_dataset, test_dataset, batch_size=128, seed=1234):
        """创建确定性的数据加载器"""

        # 训练数据加载器
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
            generator=generator,
            pin_memory=True,
        )

        # 测试数据加载器
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 确定性训练')
    parser.add_argument('--lr', default=0.1, type=float, help='学习率')
    parser.add_argument('--resume', '-r', action='store_true', help='从检查点恢复')
    parser.add_argument('--resume_path', default='./checkpoint/full_state_005.pth',
                        type=str, help='完整状态检查点路径')
    parser.add_argument('--epochs', default=200, type=int, help='总训练轮数')
    parser.add_argument('--seed', default=1234, type=int, help='随机种子')
    parser.add_argument('--batch_size', default=128, type=int, help='批次大小')
    args = parser.parse_args()

    # 1. 设置确定性训练
    setup_deterministic_training(args.seed)

    # 2. 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 3. 数据
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

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # 4. 创建确定性的数据加载器
    trainloader, testloader = DeterministicDataLoader.create(
        trainset, testset, args.batch_size, args.seed
    )

    # 5. 模型
    from modeling.backbones.model_generator import fetch_model

    class Baseline(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.in_planes = 512
            self.base = nn.Sequential(*list(fetch_model('resnet18').children())[:-2])
            self.num_classes = num_classes
            self.classifier = nn.Linear(self.in_planes, self.num_classes)

            # 设置Dropout为确定性模式
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0  # 或设置为0，避免随机性

                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = True
                    m.momentum = 0.1

        def forward(self, x):
            global_feat = self.base(x)
            global_feat = global_feat.view(global_feat.shape[0], -1)
            return self.classifier(global_feat)

    model = Baseline()

    # 6. 损失函数、优化器、调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 7. 训练状态
    start_epoch = 0
    best_acc = 0.0

    # 8. 恢复完整训练状态
    if args.resume and os.path.exists(args.resume_path):
        print(f"\n==> 从完整状态恢复: {args.resume_path}")
        start_epoch, best_acc = TrainingStateSaver.load_full_state(
            args.resume_path, model, optimizer, scheduler, trainloader, device
        )
        start_epoch += 1  # 从下一个epoch开始
    else:
        model = model.to(device)
        print("\n==> 从头开始训练")

    # 9. 训练循环
    for epoch in range(start_epoch, args.epochs):
        # 设置为训练模式
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 训练
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪（可选，增加确定性）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(trainloader)

        # 测试
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = correct / total
        avg_test_loss = test_loss / len(testloader)

        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # 更新最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc

        # 打印进度
        print(f"Epoch: {epoch + 1:03d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc * 100:.2f}% | "
              f"LR: {current_lr:.6f}")

        # 保存完整状态（每5个epoch）
        if (epoch + 1) % 5 == 0:
            state_path = f'./checkpoint/full_state_{str(epoch + 1).zfill(3)}.pth'
            TrainingStateSaver.save_full_state(
                epoch, model, optimizer, scheduler, trainloader,
                testloader, criterion, best_acc, state_path
            )

    print(f"\n训练完成! 最佳准确率: {best_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
