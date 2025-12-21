import torch
from transformers import AutoModel

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-4B-Instruct", dtype="auto", device_map="auto"
# )

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

model = AutoModel.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto",
)

with open("model_params_2.5_7b.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(name + "\n")