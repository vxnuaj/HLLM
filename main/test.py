import os
import sys
import torch
import torch.nn.functional as F
from torchinfo import summary

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import LLaMA
from config import get_config

root_path = 'main/configs'

model_config = get_config(
    root_path = root_path,
    config_type = 'model'
)

print('initializing model')

model = LLaMA(**model_config)

print('Testing Training Loss')

X = torch.load('data/tensors/train/X/X_train_0.pt')
y = torch.load('data/tensors/train/y/y_train_0.pt')

logits = model(X)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
print(f"Training Loss: {loss.item()}\n")

print('Testing Validation Loss')

X = torch.load('data/tensors/val/X/X_val.pt')
y = torch.load('data/tensors/val/y/y_val.pt')

X = X[:, :512]
y = y[:, :512]

logits = model(X)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
print(f"Validation Loss: {loss.item()}\n")