from .bases import BaseImageDataset


class ConcatDataset(BaseImageDataset):
    def __init__(self, num_class=0, train=None, test=None, domain_id=1):
        self.num_class = num_class
        self.train = train
        self.test = test
        self.domain_id = domain_id
