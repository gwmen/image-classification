# https://huggingface.co/IDEA-Research/grounding-dino-base/tree/main
import os
# DFN5B-CLIP-ViT-H-14-378
cache_vars = ['HF_HOME', 'HF_HOME', 'HF_DATASETS_CACHE']
for var in cache_vars:
    os.environ[var] = './huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # with hf-mirror
from PIL import Image
import requests
import requests

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
# text = "a cat. a remote control."
# 尝试通用的物体词汇
# text = "object . thing . item"
# 或者
# text = "person . vehicle . animal . food . furniture . electronic device"
text = "cat. remote control. food. bed"
inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)
print(results)
