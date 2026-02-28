# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys
import os

# 获取项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.env_set import seed_everything
import argparse

import torch
# from torch.backends import cudnn
#
# sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model, build_teacher_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes, num_domain = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes, num_domain)
    # teacher_model = build_teacher_model()
    teacher_model = None
    print(f'class num is : {num_classes}')
    print(f'domain num is : {num_domain}')
    if cfg.MODEL.IF_WITH_CENTER == 'no':
        # print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(cfg, num_classes)  # modified by gu
        start_epoch = 0
        scheduler = None
        # Add for using self trained model
        if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_step = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1].lstrip('0'))
            start_epoch = start_step // len(train_loader)
            print('Start epoch:', start_epoch)
            # path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            # print('Path to the checkpoint of optimizer:', path_to_optimizer)
            trained_model = torch.load(cfg.MODEL.PRETRAIN_PATH)
            model.load_state_dict(trained_model['model'])

            optimizer.load_state_dict(trained_model['optimizer'])
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            scheduler.load_state_dict(trained_model['scheduler'])

        elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
            start_epoch = 0
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
        else:
            print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

        arguments = {}

        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,  # modify for using self trained model
            loss_func,
            num_query,
            start_epoch,  # add for using self trained model
            teacher_model
        )


def main():
    parser = argparse.ArgumentParser(description="Image-classification Baseline Training")
    parser.add_argument(
        "--config_file", default="../configs/baseline-resnet50-fpn.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("fine_grained", output_dir, 0, cfg)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # new add by gu
    # cudnn.benchmark = True # changed 4 close random
    train(cfg)


if __name__ == '__main__':
    seed_everything()
    main()
# domain num is : -1
# domain num is : -1
# domain num is : -1
# domain num is : -1
#  domain num is : -1
# class num is : 200
# domain num is : -1
# 2026-02-28 14:43:33,628 fine_grained.train INFO: Start training
# 2026-02-28 14:43:46,474 fine_grained.train INFO: Epoch[1] Iteration[20/2997] Loss: 5.467, Acc: 0.000, Base Lr: 3.50e-06
# 2026-02-28 14:43:53,202 fine_grained.train INFO: Epoch[1] Iteration[40/2997] Loss: 5.419, Acc: 0.000, Base Lr: 3.50e-06
# 2026-02-28 14:43:59,949 fine_grained.train INFO: Epoch[1] Iteration[60/2997] Loss: 5.381, Acc: 0.000, Base Lr: 3.50e-06
# 2026-02-28 14:44:06,674 fine_grained.train INFO: Epoch[1] Iteration[80/2997] Loss: 5.345, Acc: 0.009, Base Lr: 3.50e-06
# 2026-02-28 14:44:13,403 fine_grained.train INFO: Epoch[1] Iteration[100/2997] Loss: 5.336, Acc: 0.006, Base Lr: 3.50e-06
# 2026-02-28 14:44:20,119 fine_grained.train INFO: Epoch[1] Iteration[120/2997] Loss: 5.339, Acc: 0.004, Base Lr: 3.50e-06
# 2026-02-28 14:44:26,905 fine_grained.train INFO: Epoch[1] Iteration[140/2997] Loss: 5.330, Acc: 0.003, Base Lr: 3.50e-06
# 2026-02-28 14:44:33,673 fine_grained.train INFO: Epoch[1] Iteration[160/2997] Loss: 5.306, Acc: 0.002, Base Lr: 3.50e-06
# 2026-02-28 14:44:40,550 fine_grained.train INFO: Epoch[1] Iteration[180/2997] Loss: 5.315, Acc: 0.001, Base Lr: 3.50e-06
# 2026-02-28 14:44:47,310 fine_grained.train INFO: Epoch[1] Iteration[200/2997] Loss: 5.310, Acc: 0.001, Base Lr: 3.50e-06
# 2026-02-28 14:44:54,051 fine_grained.train INFO: Epoch[1] Iteration[220/2997] Loss: 5.315, Acc: 0.010, Base Lr: 3.50e-06
# 2026-02-28 14:45:00,771 fine_grained.train INFO: Epoch[1] Iteration[240/2997] Loss: 5.306, Acc: 0.007, Base Lr: 3.50e-06
# 2026-02-28 14:45:07,521 fine_grained.train INFO: Epoch[1] Iteration[260/2997] Loss: 5.297, Acc: 0.014, Base Lr: 3.50e-06
# 2026-02-28 14:45:14,253 fine_grained.train INFO: Epoch[1] Iteration[280/2997] Loss: 5.305, Acc: 0.016, Base Lr: 3.50e-06
# 2026-02-28 14:45:20,991 fine_grained.train INFO: Epoch[1] Iteration[300/2997] Loss: 5.305, Acc: 0.021, Base Lr: 3.50e-06
# 2026-02-28 14:45:27,727 fine_grained.train INFO: Epoch[1] Iteration[320/2997] Loss: 5.295, Acc: 0.014, Base Lr: 3.50e-06
# 2026-02-28 14:45:34,482 fine_grained.train INFO: Epoch[1] Iteration[340/2997] Loss: 5.292, Acc: 0.009, Base Lr: 3.50e-06
# 2026-02-28 14:45:41,259 fine_grained.train INFO: Epoch[1] Iteration[360/2997] Loss: 5.309, Acc: 0.006, Base Lr: 3.50e-06
# 2026-02-28 14:45:48,054 fine_grained.train INFO: Epoch[1] Iteration[380/2997] Loss: 5.325, Acc: 0.004, Base Lr: 3.50e-06
# 2026-02-28 14:45:54,789 fine_grained.train INFO: Epoch[1] Iteration[400/2997] Loss: 5.345, Acc: 0.003, Base Lr: 3.50e-06
# 2026-02-28 14:46:01,522 fine_grained.train INFO: Epoch[1] Iteration[420/2997] Loss: 5.329, Acc: 0.002, Base Lr: 3.50e-06
# 2026-02-28 14:46:08,272 fine_grained.train INFO: Epoch[1] Iteration[440/2997] Loss: 5.313, Acc: 0.001, Base Lr: 3.50e-06
# 2026-02-28 14:46:15,012 fine_grained.train INFO: Epoch[1] Iteration[460/2997] Loss: 5.322, Acc: 0.001, Base Lr: 3.50e-06
# 2026-02-28 14:46:21,788 fine_grained.train INFO: Epoch[1] Iteration[480/2997] Loss: 5.327, Acc: 0.001, Base Lr: 3.50e-06
# 2026-02-28 14:46:28,537 fine_grained.train INFO: Epoch[1] Iteration[500/2997] Loss: 5.315, Acc: 0.000, Base Lr: 3.50e-06
# 2026-02-28 14:46:35,301 fine_grained.train INFO: Epoch[1] Iteration[520/2997] Loss: 5.319, Acc: 0.000, Base Lr: 3.50e-06