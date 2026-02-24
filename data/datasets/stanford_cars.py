# encoding: utf-8
"""
@author:  sherlock

"""

import glob
import os
import re

import os.path as osp

try:
    from .bases import BaseImageDataset
except Exception as e:
    print(e)  # just for debug
    from data.datasets.bases import BaseImageDataset


# down from
# https://www.modelscope.cn/datasets/iic/foundation_model_evaluation_benchmark/files
class StanfordCars(BaseImageDataset):
    dataset_dir = 'Stanford-Cars'
    class_set = None
    class_dict = dict()

    def __init__(self, root='/home/data', start_id=0, verbose=True, **kwargs):
        super(StanfordCars, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.start_id = start_id

        self.train_dir = osp.join(self.dataset_dir, 'images_train')
        self.test_dir = osp.join(self.dataset_dir, 'images_test')

        self.num_class = None
        self.domain_id = -1
        self.train = self._process(self.train_dir)
        self.test = self._process(self.test_dir)

    def _process(self, input_dir):
        # (img_path, clss id)
        _info = list()
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
            [_info.append((o, self.class_dict[_class], -1)) for o in details]  # path, class id, domain id

        return _info


if __name__ == '__main__':
    oo = StanfordCars(root=r'E:\20260130\dataset')
    print()
