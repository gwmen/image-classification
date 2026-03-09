# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
from PIL import Image
from .transforms import RandomErasing


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize((cfg.INPUT.SIZE_RESIZE, cfg.INPUT.SIZE_RESIZE), Image.BILINEAR),
            T.RandomCrop((cfg.INPUT.SIZE_TRAIN, cfg.INPUT.SIZE_TRAIN)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
            T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            T.ToTensor(),
            normalize_transform
        ])
    else:
        transform = T.Compose([
            T.Resize((cfg.INPUT.SIZE_RESIZE, cfg.INPUT.SIZE_RESIZE), Image.BILINEAR),
            T.CenterCrop((cfg.INPUT.SIZE_TEST, cfg.INPUT.SIZE_TEST)),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
