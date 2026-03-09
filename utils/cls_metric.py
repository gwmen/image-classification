# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric


class ClassificationMetric(Metric):
    # Top-5 accuracy
    def __init__(self, num_query=0, max_rank=50, feat_norm='yes'):
        super(ClassificationMetric, self).__init__()
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
        sum_corrects_1, sum_corrects_3, sum_corrects_5 = 0, 0, 0
        total_sample = 0
        for label_id, predicted in zip(self.labels, self.predicted):
            _, predicted_top_k = torch.topk(predicted, 5)
            correct_1 = predicted_top_k[:, 0].eq(label_id).sum().item()
            correct_2 = predicted_top_k[:, 1].eq(label_id).sum().item()
            correct_3 = predicted_top_k[:, 2].eq(label_id).sum().item()
            correct_4 = predicted_top_k[:, 3].eq(label_id).sum().item()
            correct_5 = predicted_top_k[:, 4].eq(label_id).sum().item()
            total_sample += len(label_id)
            sum_corrects_1 += correct_1
            sum_corrects_3 += (correct_1 + correct_2 + correct_3)
            sum_corrects_5 += (correct_1 + correct_2 + correct_3 + correct_4 + correct_5)
        top_1 = sum_corrects_1 / total_sample
        top_3 = sum_corrects_3 / total_sample
        top_5 = sum_corrects_5 / total_sample
        return top_1, top_3, top_5
