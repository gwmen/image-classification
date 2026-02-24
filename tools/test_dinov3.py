import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers.image_utils import load_image
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoImageProcessor, AutoModel

dino_root = '../../'
selected_model_path = os.path.join(dino_root, 'dinov3-vits16-pretrain-lvd1689m')


def compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord):
    """
    计算指定patch与其他所有patch的余弦相似性，并生成热力图

    Args:
        patch_features: patch特征张量, shape (1, num_patches, feature_dim)
        H: patch网格高度
        W: patch网格宽度
        target_patch_coord: 目标patch坐标 (h_idx, w_idx)

    Returns:
        heatmap: 相似性热力图, shape (H, W)
    """

    # 检查输入特征数量是否与网格尺寸匹配
    assert patch_features.shape[1] == H * W, f"特征数量{H * W}与网格大小{H}x{W}不匹配"

    # 提取目标patch的特征
    target_idx = target_patch_coord[0] * W + target_patch_coord[1]
    target_feature = patch_features[0, target_idx]  # shape (feature_dim,)

    # 使用余弦相似性计算目标特征与所有patch特征之间的相似度
    similarities = F.cosine_similarity(
        target_feature.unsqueeze(0),  # shape (1, feature_dim)
        patch_features[0],  # shape (num_patches, feature_dim)
        dim=1
    )

    # 将一维相似性向量重塑为二维热力图
    heatmap = similarities.reshape(H, W).cpu().numpy()

    return heatmap


def plot_similarity_heatmap(heatmap, target_patch_coord):
    """
    绘制相似性热力图，并在目标patch位置显示红点
    Args:
        heatmap: 相似性热力图, shape (H, W)
        target_patch_coord: 目标patch坐标 (h_idx, w_idx)
    """
    H, W = heatmap.shape

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 显示热力图
    im = ax.imshow(heatmap, cmap='viridis', aspect='equal')

    # 在目标patch位置添加红点
    target_h, target_w = target_patch_coord
    ax.plot(target_w, target_h, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)

    # 添加颜色条，使其与图像高度对齐
    cbar = plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.7)
    cbar.ax.tick_params(labelsize=8)  # 调整颜色条刻度标签大小

    # 设置坐标轴标签和标题
    ax.set_xlabel('Width (patch index)')
    ax.set_ylabel('Height (patch index)')
    ax.set_title(f'Cosine Similarity to Patch at ({target_h}, {target_w})')

    # 设置网格线
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    # ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # 设置主刻度
    ax.set_xticks(np.arange(0, W, max(1, W // 10)))
    ax.set_yticks(np.arange(0, H, max(1, H // 10)))
    # 调整刻度标签大小
    ax.tick_params(axis='both', which='major', labelsize=8)

    plt.tight_layout()
    plt.show()
    return fig, ax


url = r'test1.jpg'
image = load_image(url)

processor = AutoImageProcessor.from_pretrained(selected_model_path,
                                               size=224,  # 覆盖默认尺寸
                                               do_resize=True,
                                               do_center_crop=False  # 禁用中心裁剪，保持完整图像
                                               )
model = AutoModel.from_pretrained(selected_model_path).cuda()
inputs = processor(images=image, return_tensors="pt").to(model.device)

#
with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)
    #            'layer1': teacher_features.hidden_states[3],  # 浅层特征
    #                 'layer2': teacher_features.hidden_states[6],
    #                 'layer3': teacher_features.hidden_states[12],
    #                 'layer4': teacher_features.hidden_states[24], # 深层特征
    # pooler_output = outputs.pooler_output  # [batch, hidden_size]
    cls = outputs.last_hidden_state[:, 0]  # 全局（[CLS]）
    num_regs = model.config.num_register_tokens
    #  1 class token + 4 register tokens + 196 patch tokens = 201 tokens
    patch_flat = outputs.last_hidden_state[:, 1 + num_regs:, :]  # ← 这就是40×40的patches!

    B, N, C = patch_flat.shape
    H = W = int((N) ** 0.5)

    patch_features = patch_flat  # outputs.last_hidden_state[:, 1:, :]  # 去掉CLS token
    print("patch_features.shape: ", patch_features.shape)

    print("(H, W):", H, W)

    target_w = 4
    target_h = 6

    target_patch_coord = (target_h, target_w)

    heatmap = compute_patch_similarity_heatmap(patch_features, H, W, target_patch_coord)
    plot_similarity_heatmap(heatmap, target_patch_coord)

# https://blog.csdn.net/weixin_39806242/article/details/154171081

# https://blog.csdn.net/weixin_52613525/article/details/152179330
