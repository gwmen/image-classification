import torch
import random
import numpy as np
import os
import sys

import torch

sys.path.append('.')
_target = '.cache'
_py_hash_seed_env = "PYTHONHASHSEED"
cache_vars = ['HF_HOME', 'HF_HOME', 'HF_DATASETS_CACHE']
for var in cache_vars:
    os.environ[var] = f'../{_target}/huggingface'

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # with hf-mirror
os.environ['TORCH_HOME'] = f'../{_target}/torch_hub'


def seed_everything(seed=1234):
    """
    Ensure consistent repetition of the experiment
    """
    random.seed(seed)
    os.environ[_py_hash_seed_env] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
