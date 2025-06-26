import os
import sys
import torch

from torchinfo import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import LLaMA
from config import get_config

root_path = 'main/configs'

model_config = get_config(
    root_path = root_path,
    config_type = 'model'
)

model = LLaMA(**model_config)    

x = torch.randint(low = 0, high = 10000, size = (1, 512)) # batch_size \times seq_len

summary(model, input_data = x)