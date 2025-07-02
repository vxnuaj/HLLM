import torch
from tqdm import tqdm
import os
import sys

'''
print(f"Number of X tensors: {len(os.listdir('data/tensors/train/X'))}")
print(f"Number of y tensors: {len(os.listdir('data/tensors/train/y'))}")

X_samples = 0
y_samples = 0
X_toks = 0
y_toks = 0
X_val_toks = 0
y_val_toks = 0

print(f"Getting Training Tokens")

for i in tqdm(range(len(os.listdir('data/tensors/train/X')))):
    X = torch.load(f'data/tensors/train/X/X_train_{i}.pt')
    y = torch.load(f'data/tensors/train/y/y_train_{i}.pt')

    X_toks += X.shape[1] * X.shape[0]
    y_toks += y.shape[1] * X.shape[0]

    del X, y

assert X_toks == y_toks, ValueError('Number of X tokens does not match the number of y tokens')

print(X_toks)
print(y_toks)

print('Getting Val Tokens')

for i in tqdm(range(len(os.listdir('data/tensors/val/X')))):
    X = torch.load(f'data/tensors/val/X/X_val.pt')
    y = torch.load(f'data/tensors/val/y/Y_val.pt')

    X_val_toks += X.shape[1] * X.shape[0]
    y_val_toks += y.shape[1] * X.shape[0]

    del X, y

assert X_toks == y_toks, ValueError('Number of X tokens does not match the number of y tokens')

print(X_val_toks)

print(f"Total Tokens: {X_toks + X_val_toks}")'''

X_val = torch.load('data/tensors/train/X/X_train_1.pt')
y_val = torch.load('data/tensors/train/y/y_train_0.pt')

print(X_val[0])
print(y_val[0])