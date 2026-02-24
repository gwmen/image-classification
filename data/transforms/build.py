# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


#         transform = transforms.Compose([
#             transforms.Resize((256, 256)),
#             transforms.RandomRotation(15),
#             transforms.RandomResizedCrop((224, 224)),
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.ColorJitter(
#                 brightness=0.2,
#                 contrast=0.2,
#                 saturation=.2,
#                 hue=.1
#             ),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean = [0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])
def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        # transform = T.Compose([
        #     T.Resize(cfg.INPUT.SIZE_TRAIN),
        #     T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        #     T.Pad(cfg.INPUT.PADDING),
        #     T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        #     T.ToTensor(),
        #     normalize_transform,
        #     RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        # ])
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomRotation(15),
            T.RandomResizedCrop(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=.2,
                hue=.1
            ),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
