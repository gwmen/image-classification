# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from utils.env_set import seed_everything
import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
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
        mkdir(output_dir)

    # logger = setup_logger("reid_baseline", output_dir, 0)
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
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes, _ = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    model.load_param(
        r"E:\20260130\fine-gtrained-cv\output\Fine-grained-baseline\resnet18\resnet50_checkpoint_00000392.pt")

    inference(cfg, model, val_loader, num_query)


if __name__ == '__main__':
    main()
# 2026-02-11 16:47:48,201 fine_grained.inference INFO: Enter inferencing
# 2026-02-11 16:48:31,450 fine_grained.inference INFO: top_1: 52.84%
# 2026-02-11 16:48:31,450 fine_grained.inference INFO: top_2: 68.43%
# 2026-02-11 16:48:31,450 fine_grained.inference INFO: top_3: 77.17%