# encoding: utf-8
"""
@author:  sherlock

"""

import glob
import os
import random
import re

import os.path as osp

try:
    from .bases import BaseImageDataset
except Exception as e:
    print(e)  # just for debug
    from data.datasets.bases import BaseImageDataset


# down from
# ！ -----change https://www.modelscope.cn/datasets/iic/foundation_model_evaluation_benchmark/files-------
# USING https://github.com/chou141253/FGVC-PIM
class CUB2002011(BaseImageDataset):
    dataset_dir = 'cub200/datas'
    class_dict = dict()

    def __init__(self, root='/home/data', start_id=0, verbose=True, **kwargs):
        super(CUB2002011, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.start_id = start_id
        self.num_class = None
        self.domain_id = -1
        self.train = list()
        self.test = list()
        self._process(self.dataset_dir)

    def _process(self, input_dir):
        _phase = ['train', 'test']
        train_classes = os.listdir(os.path.join(input_dir, _phase[0]))
        self.num_class = len(train_classes)
        [self.class_dict.update({o.split('.')[0].lstrip('0'): o.split('.')[-1]}) for o in train_classes]
        for _p in _phase:
            cur_p_list = getattr(self, _p)
            classes = os.listdir(os.path.join(input_dir, _p))
            for cls in classes:
                cls_id, cls_name = cls.split('.')
                cls_id = int(cls_id) - 1
                cur_dir = os.path.join(input_dir, _p, cls)
                cur_imgs = os.listdir(cur_dir)
                [cur_p_list.append((os.path.join(cur_dir, o), cls_id, -1)) for o in cur_imgs]


if __name__ == '__main__':
    oo = CUB2002011(root=r'E:\20260130\FGVC Good job\dataset')
    print()
