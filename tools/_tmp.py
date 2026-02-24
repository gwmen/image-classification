import torch
import torchvision
from torchvision.models import ResNet50_Weights

# 方法1A：使用最新的预训练权重
model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)