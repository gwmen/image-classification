# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.cls_metric import ClsMetric


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            # data, pids, camids = batch
            # data = data.to(device) if torch.cuda.device_count() >= 1 else data
            data, target, target_domain = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            target = target.to(device) if torch.cuda.device_count() >= 1 else target
            predict, _ = model(data)
            return predict, target

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("fine_grained.inference")
    logger.info("Enter inferencing")

    evaluator = create_supervised_evaluator(model, metrics={
        'top_k': ClsMetric(),
    }, device=device)
    evaluator.run(val_loader)
    top_1, top_2, top_3 = evaluator.state.metrics['top_k']
    logger.info("top_1: {:.2%}".format(top_1))
    logger.info("top_2: {:.2%}".format(top_2))
    logger.info("top_3: {:.2%}".format(top_3))
