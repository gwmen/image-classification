# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import domain_collate_fn
from .datasets import init_dataset, ImageDataset, concat_datasets
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = concat_datasets([init_dataset(_name, root=cfg.DATASETS.ROOT_DIR) for _name in cfg.DATASETS.NAMES])
    num_classes = dataset.num_class
    num_domain = dataset.domain_id
    train_set = ImageDataset(dataset.train, train_transforms)
    val_set = ImageDataset(dataset.test, val_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=domain_collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=domain_collate_fn  # collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.test), num_classes, num_domain
