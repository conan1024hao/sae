import unittest

import torch
import torch.nn as nn
from peft import inject_adapter_in_model

from sae import TopKSaeConfig, get_peft_sae_model


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

model = DummyModel()
config = TopKSaeConfig(k=1, num_latents=5, target_modules=["linear"])

for name, param in model.named_parameters():
    print(name)

# Inject the adapter into the model
model = inject_adapter_in_model(config, model)

print("After injecting adapter:")
for name, param in model.named_parameters():
    print(name)
