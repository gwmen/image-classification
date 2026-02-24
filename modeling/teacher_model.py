import os
from transformers import AutoModel

dino_root = '../../'


def get_dino_v3():
    selected_model_path = os.path.join(dino_root, 'dinov3-vits16-pretrain-lvd1689m')
    return AutoModel.from_pretrained(selected_model_path).cuda()
