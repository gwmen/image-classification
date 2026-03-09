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
    num_workers = cfg.INPUT.NUM_WORKERS
    dataset = init_dataset(cfg.INPUT.DATASET, root=cfg.INPUT.ROOT_DIR)
    num_classes = dataset.num_class
    num_domain = dataset.domain_id
    train_set = ImageDataset(dataset.train, train_transforms)
    val_set = ImageDataset(dataset.test, val_transforms)

    train_loader = DataLoader(
        train_set, batch_size=cfg.INPUT.TRAIN_BATCH, shuffle=True, num_workers=num_workers,
        # sampler=None,
        collate_fn=domain_collate_fn
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.INPUT.TEST_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=domain_collate_fn  # collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.test), num_classes, num_domain
