import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFocalLoss(nn.Module):
    """
    https://github.com/artemie220284-stack/ResNet-Image-Recognition-Suite/tree/master/ResNet18/focal_loss_continue.ipynb
    为了使用 model.to(device)自动管理设备，虽然这段代码里没有可训练参数，仍然使用 nn.Module。
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean', adaptive_gamma=True):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.base_gamma = gamma
        self.adaptive_gamma = adaptive_gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.adaptive_gamma and self.training:
            with torch.no_grad():
                _, predicted = inputs.max(1)
                acc = predicted.eq(targets).float().mean().item()  # 准确率越高，越关注难样本

                dynamic_gamma = self.base_gamma * (1.0 + (1.0 - acc) * 0.5)
        else:
            dynamic_gamma = self.base_gamma

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** dynamic_gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class AdaptiveFocalLoss(nn.Module):
#     """
#     适用于多分类任务 (Softmax + CrossEntropy)
#     支持动态 Gamma 调整
#     """
#
#     def __init__(self, alpha=None, base_gamma=2.0, reduction='mean', adaptive_gamma=True):
#         super(AdaptiveFocalLoss, self).__init__()
#         self.alpha = alpha  # 类别权重 Tensor
#         self.base_gamma = base_gamma
#         self.reduction = reduction
#         self.adaptive_gamma = adaptive_gamma
#
#     def forward(self, inputs, targets):
#         # 1. 动态调整 Gamma
#         if self.adaptive_gamma and self.training:
#             with torch.no_grad():
#                 _, predicted = inputs.max(1)
#                 acc = predicted.eq(targets).float().mean().item()
#                 # 准确率越高，gamma 越大（越关注难样本）
#                 dynamic_gamma = self.base_gamma * (1.0 + (1.0 - acc) * 0.5)
#         else:
#             dynamic_gamma = self.base_gamma
#
#         # 2. 计算 Focal Loss (数值稳定版本)
#         log_softmax = F.log_softmax(inputs, dim=1)
#         ce_loss = F.nll_loss(log_softmax, targets, reduction='none', weight=self.alpha)
#
#         # 获取真实类别的概率 p_t
#         pt = torch.exp(-ce_loss)  # 因为 ce_loss = -log(p_t)，所以 p_t = exp(-ce_loss)
#
#         focal_loss = (1 - pt) ** dynamic_gamma * ce_loss
#
#         # 3. Reduction
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
