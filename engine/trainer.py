# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.cls_metric import ClsMetric

global ITER
ITER = 0


def _infer_teacher(ims_tensor, teacher_model):
    out = teacher_model(ims_tensor)
    num_regs = teacher_model.config.num_register_tokens
    patch_flat = out.last_hidden_state[:, 1 + num_regs:, :]  # 1 class token + 4 register tokens + num patch tokens
    _, nb, _ = patch_flat.shape
    nh = nw = int(nb ** 0.5)
    pool_out = out.pooler_output
    fea = patch_flat.view(len(ims_tensor), nh, nw, 384).permute(0, 3, 1, 2)  # B P C->B H W C ->B C H W
    return fea, pool_out


def _distillation(stu_fea, tea_fea):
    criterion = nn.KLDivLoss(reduction='mean')  # 'batch mean' ?
    return criterion(torch.log_softmax(stu_fea, dim=-1), torch.softmax(tea_fea, dim=-1))


def create_supervised_trainer(model, optimizer, loss_fn,
                              teacher_model=None, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        teacher_model

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        # https://runebook.dev/zh/docs/pytorch/generated/torch.optim.sgd/torch.optim.SGD.load_state_dict
        # for state in optimizer.state.values():
        #     for key, value in state.items():
        #         if isinstance(value, torch.Tensor):
        #             state[key] = value.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target, target_domain = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target

        score, stu_feat = model(img)
        loss = loss_fn(score, target)
        if teacher_model:
            with torch.inference_mode():
                tea_fea, fea_cls = _infer_teacher(img, teacher_model)
            loss = .5 * loss + .5 * _distillation(stu_feat, tea_fea)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


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


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        teacher_model
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("fine_grained.train")
    # logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, teacher_model, device=device)
    evaluator = create_supervised_evaluator(model, metrics={
        # 'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM),
        'top_k': ClsMetric(),
        # 'precision': Precision(average=True),
        # 'recall': Recall(average=True),
        # 'loss': Loss(loss_fn)  # criterion为损失函数
    }, device=device)
    # checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    checkpointer = ModelCheckpoint(output_dir,
                                   n_saved=cfg.SOLVER.MAX_EPOCHS + 2,
                                   filename_pattern=cfg.MODEL.NAME + '_checkpoint_' + '{global_step:08d}.pt',
                                   require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'scheduler': scheduler})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            top_1, top_2, top_3 = evaluator.state.metrics['top_k']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("top_1: {:.2%}".format(top_1))
            logger.info("top_2: {:.2%}".format(top_2))
            logger.info("top_3: {:.2%}".format(top_3))

    trainer.run(train_loader, max_epochs=epochs)
