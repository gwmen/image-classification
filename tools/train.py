# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.env_set import seed_everything
import argparse
import torch

from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger


def train(config):
    # prepare dataset
    train_loader, val_loader, _, num_classes, _ = make_data_loader(config)

    # prepare model
    model = build_model(config, num_classes)

    print(f'class num is : {num_classes}')

    optimizer = make_optimizer(config, model)
    loss_func = make_loss(config, num_classes)
    start_epoch = 0

    if os.path.isfile(config.MODEL.RESUME_PATH):
        trained_model = torch.load(config.MODEL.RESUME_PATH)
        model.load_state_dict(trained_model['model'])
        optimizer.load_state_dict(trained_model['optimizer'])
        scheduler = WarmupMultiStepLR(optimizer, config.SOLVER.STEPS, config.SOLVER.GAMMA, config.SOLVER.WARMUP_FACTOR,
                                      config.SOLVER.WARMUP_ITERS, config.SOLVER.WARMUP_METHOD)
        scheduler.load_state_dict(trained_model['scheduler'])
    else:
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, config.SOLVER.STEPS, config.SOLVER.GAMMA, config.SOLVER.WARMUP_FACTOR,
                                      config.SOLVER.WARMUP_ITERS, config.SOLVER.WARMUP_METHOD)

    arguments = {}

    do_train(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        start_epoch,
        arguments
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-grained Image classification Baseline Training")
    parser.add_argument(
        "--config_file", default="../configs/cub200-res50-softmax.yml", help="path to config file", type=str
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
    # logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        # with open(args.config_file, 'r') as cf:
        #     config_str = "\n" + cf.read()
        #     logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train(cfg)


if __name__ == '__main__':
    seed_everything()
    main()
