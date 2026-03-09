# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .teacher_model import get_dino_v3


def build_model(cfg, num_classes, num_domain=-1):
    arguments = {}
    model = Baseline(cfg.MODEL.NAME, cfg.MODEL.HEAD, num_classes)
    return model


def build_teacher_model():
    return get_dino_v3()
