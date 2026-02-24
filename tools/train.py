# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from utils.env_set import seed_everything
import argparse
import os
import sys

import torch
# from torch.backends import cudnn
#
# sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model, build_teacher_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    train_loader, val_loader, num_query, num_classes, num_domain = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes, num_domain)
    teacher_model = build_teacher_model()
    print(f'class num is : {num_classes}')
    print(f'domain num is : {num_domain}')
    if cfg.MODEL.IF_WITH_CENTER == 'no':
        print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
        optimizer = make_optimizer(cfg, model)
        # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
        #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

        loss_func = make_loss(cfg, num_classes)  # modified by gu

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
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="../configs/baseline.yml", help="path to config file", type=str
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

    logger = setup_logger("fine_grained", output_dir, 0)
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
