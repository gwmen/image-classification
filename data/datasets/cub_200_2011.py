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
# https://www.modelscope.cn/datasets/iic/foundation_model_evaluation_benchmark/files
class CUB2002011(BaseImageDataset):
    dataset_dir = 'CUB-200-2011/Images'
    class_set = None
    class_dict = dict()
    _r = 2 / 3
    _train_txt = 'train.txt'
    _test_txt = 'test.txt'

    def __init__(self, root='/home/data', start_id=0, verbose=True, **kwargs):
        super(CUB2002011, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_txt = osp.join(root, self._train_txt) # 一次性生成，避免以后随机划分出问题
        self.test_txt = osp.join(root, self._test_txt)
        self.start_id = start_id
        self.num_class = None
        self.domain_id = -1
        self.train = list()
        self.test = list()
        self._process(self.dataset_dir)

    def _process(self, input_dir):
        # (img_path, clss id)
        # _info = list()
        _classes = os.listdir(input_dir)
        if self.class_set:
            assert self.class_set == set(_classes)
        else:
            self.class_set = set(_classes)
            self.num_class = len(_classes)
            [self.class_dict.update({o: int(i + self.start_id)}) for i, o in enumerate(_classes)]
        for _class in _classes:
            _path = os.path.join(input_dir, _class)
            details = [os.path.join(_path, o) for o in os.listdir(_path)]
            random.shuffle(details)
            bound_id = int(len(details) * self._r)
            [self.train.append((o, self.class_dict[_class], -1))  # path, class id, domain id
             if i < bound_id else self.test.append((o, self.class_dict[_class], -1))
             for i, o in enumerate(details)]


if __name__ == '__main__':
    oo = CUB2002011(root=r'E:\20260130\dataset')
    print()
