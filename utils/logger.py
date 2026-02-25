# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name, save_dir, distributed_rank, exp_info=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:  # 防止重复添加 handler
        return logger

    logger.propagate = False  # 防止日志向上传播到 root logger
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    model = exp_info.MODEL.NAME
    loss_fn = exp_info.MODEL.METRIC_LOSS_TYPE
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        current_time = datetime.now().strftime("%Y%m%d_%H-%M-%S")
        exp_name = f"log_{current_time}_{model}_{loss_fn}.txt"
        fh = logging.FileHandler(os.path.join(save_dir, exp_name), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
