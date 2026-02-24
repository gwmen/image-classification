import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm

# 查看所有 SE-ResNeXt 相关模型
print("可用的 SE-ResNeXt 模型:")
for name in timm.list_models('*resnext*'):
    if 'se' in name.lower():
        print(f"  - {name}")
from timm import get_pretrained_cfg

# 1. 获取模型默认配置
cfg = get_pretrained_cfg("seresnext50_32x4d")

# 2. 修改配置，指向自定义 .pth URL
# cfg.url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/se_resnext50_32x4d_2020-07-01-0f36d90c.pth"
# cfg.file = "se_resnext50_32x4d_2020-07-01-0f36d90c.pth"
cfg.hf_hub_id = None  # 禁用 Hugging Face

# # 3. 注册修改后的配置
# timm.models.register_model("seresnext50_32x4d", cfg)
#
# timm.create_model("seresnext50_32x4d",pretrained=True)
url = cfg.url
model = timm.create_model(
    "seresnext50_32x4d",
    pretrained=False,
)
import torch

#
#
state_dict = torch.hub.load_state_dict_from_url(
    url,
    progress=True,  # 显示进度条
    map_location='cpu'  # 加载到 CPU
)

# 3. 加载到模型
model.load_state_dict(state_dict)
print('---')

# # 获取配置
cfg = get_pretrained_cfg("seresnext50_32x4d")

# # 修改配置
# cfg_overlay = {
#     "hf_hub_id": None
# }
#
# # 加载
# model1 = timm.create_model(
#     "seresnext50_32x4d",
#     pretrained=True,
#     pretrained_cfg_overlay=cfg_overlay
# )


# 加载
model1 = timm.create_model(
    "seresnext50_32x4d",
    pretrained=True,
)
print()

# 方法1. 加载 safe tensor
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# model1 = timm.create_model(
#     "seresnext50_32x4d",
#     pretrained=True,
# )
# 方法2. 修改配置
# # 获取配置
# cfg = get_pretrained_cfg("seresnext50_32x4d")
# # 修改配置
# cfg_overlay = {
#     "hf_hub_id": None
# }
# model1 = timm.create_model(
#     "seresnext50_32x4d",
#     pretrained=True,
#     pretrained_cfg_overlay=cfg_overlay
# )

# 方法3. torch.hub.load_state_dict_from_url 获取权重再加载
# url = cfg.url
# state_dict = torch.hub.load_state_dict_from_url(
#     url,
#     progress=True,  # 显示进度条
#     map_location='cpu'  # 加载到 CPU
# )
# model1 = timm.create_model(
#     "seresnext50_32x4d",
#     pretrained=False,
# )
# model1.load_state_dict(state_dict)
# model = nn.Sequential(*list(self.model.children())[:-3])
# # 使用 named_children 查看结构
# for name, child in self.model.named_children():
#     print(name)
#
# # 更可控的切片方式
# if isinstance(self.model, nn.Sequential):
#     layers = []
#     for i, layer in enumerate(self.model):
#         if i < len(self.model) - 3:  # 保留前 n-3 层
#             layers.append(layer)
#     model = nn.Sequential(*layers)
