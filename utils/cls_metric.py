# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric


class ClsMetric(Metric):
    def __init__(self, num_query=0, max_rank=50, feat_norm='yes'):
        super(ClsMetric, self).__init__()
        self.predicted = []
        self.labels = []

    def reset(self):
        self.predicted = []
        self.labels = []

    def update(self, output):
        predicted, cls_id = output
        self.labels.append(cls_id)
        self.predicted.append(predicted)

    def compute(self):
        sum_corrects_1, sum_corrects_2, sum_corrects_3 = 0, 0, 0
        total_sample = 0
        for label_id, predicted in zip(self.labels, self.predicted):
            _, predicted_top_k = torch.topk(predicted, 3)
            correct_1 = predicted_top_k[:, 0].eq(label_id).sum().item()
            correct_2 = predicted_top_k[:, 1].eq(label_id).sum().item()
            correct_3 = predicted_top_k[:, 2].eq(label_id).sum().item()
            total_sample += len(label_id)
            sum_corrects_1 += correct_1
            sum_corrects_2 += (correct_1 + correct_2)
            sum_corrects_3 += (correct_1 + correct_2 + correct_3)
        top_1 = sum_corrects_1 / total_sample
        top_2 = sum_corrects_2 / total_sample
        top_3 = sum_corrects_3 / total_sample
        return top_1, top_2, top_3
