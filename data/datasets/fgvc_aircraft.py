# encoding: utf-8
"""
@author:  sherlock

"""

import glob
import os
import re

import os.path as osp
import pandas as pd

# try:
#     from .bases import BaseImageDataset
# except Exception as e:
#     print(e)  # just for debug
#     from data.datasets.bases import BaseImageDataset
from .bases import BaseImageDataset


# down from
# https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
class FGVCAircraft(BaseImageDataset):
    dataset_dir = 'FGVC-Aircraft/data'
    class_set = None
    class_dict = dict()

    def __init__(self, root='/home/data', start_id=0, verbose=True, **kwargs):
        super(FGVCAircraft, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.start_id = start_id

        self.train_dir = osp.join(self.dataset_dir, 'images_variant_trainval.txt')
        self.test_dir = osp.join(self.dataset_dir, 'images_variant_test.txt')

        self.num_class = None
        self.domain_id = -1
        self.train = self._process(self.train_dir)
        self.test = self._process(self.test_dir)

    def _process(self, input_dir):
        _info = list()
        with open(input_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.rstrip('\n').split(' ') for line in lines]
        _classes = [' '.join(o[1:]) for o in lines]
        _classes = list(set(_classes))
        _classes.sort()
        if self.class_set:
            assert self.class_set == set(_classes)
        else:
            self.class_set = set(_classes)
            self.num_class = len(self.class_set)
            [self.class_dict.update({o: int(i + self.start_id)}) for i, o in enumerate(self.class_set)]
        for pic_info in lines:
            pic_name = pic_info[0]
            pic_path = os.path.join(self.dataset_dir, f'images/{pic_name}.jpg')
            _class = ' '.join(pic_info[1:])
            _info.append((pic_path, self.class_dict[_class], -1))  # path, class id, domain id

        return _info


if __name__ == '__main__':
    oo = FGVCAircraft(root=r'E:\20260130\dataset')
    print()
