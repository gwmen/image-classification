# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
# from .cuhk03 import CUHK03
# from .dukemtmcreid import DukeMTMCreID
# from .market1501 import Market1501
# from .msmt17 import MSMT17
# from .veri import VeRi
# from .cifar10 import Cifar10
from .dataset_loader import ImageDataset
from .concat_dataset import ConcatDataset
from .stanford_cars import StanfordCars
from .stanford_dogs import StanfordDogs
from .cub_200_2011 import CUB2002011
from .fgvc_aircraft import FGVCAircraft
from .na_birds import NABirds

# class StanfordDogs(BaseImageDataset):
__factory = {
    # 'market1501': Market1501,
    # # 'cuhk03': CUHK03,
    # 'dukemtmc': DukeMTMCreID,
    # 'msmt17': MSMT17,
    # 'veri': VeRi,
    # 'cifar10': Cifar10,
    'stanford_cars': StanfordCars,
    'stanford_dogs': StanfordDogs,
    'cub_200_2011': CUB2002011,
    'na_birds': NABirds,
    'fgvc_aircraft': FGVCAircraft
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)


def concat_datasets(datasets: list):
    if len(datasets) == 1:
        return datasets[0]
    else:
        num_class = 0
        train_list = list()
        test_list = list()
        for domain_id, dataset in enumerate(datasets):
            [train_list.append((o[0], o[1] + num_class, domain_id)) for o in dataset.train]
            [test_list.append((o[0], o[1] + num_class, domain_id)) for o in dataset.test]
            num_class += dataset.num_class
        return ConcatDataset(num_class=num_class, train=train_list, test=test_list, domain_id=len(datasets))
