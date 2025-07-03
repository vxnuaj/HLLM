import torch
from tqdm import tqdm
import os
import sys

print(f"Number of X tensors: {len(os.listdir('data/tensors/train/X'))}")
print(f"Number of y tensors: {len(os.listdir('data/tensors/train/y'))}")

X_samples = 0
y_samples = 0
X_toks = 0
y_toks = 0
X_seqs = 0
y_seqs = 0
X_val_toks = 0
y_val_toks = 0
X_val_seqs = 0
y_val_seqs = 0

print(f"Getting Training Stats")

for i in tqdm(range(len(os.listdir('data/tensors/train/X')))):
    X = torch.load(f'data/tensors/train/X/X_train_{i}.pt')
    y = torch.load(f'data/tensors/train/y/y_train_{i}.pt')

    X_toks += X.shape[1] * X.shape[0]
    y_toks += y.shape[1] * X.shape[0]
    X_seqs += X.shape[0]
    y_seqs += y.shape[0]

    del X, y

assert X_toks == y_toks, ValueError('Number of X tokens does not match the number of y tokens')
assert X_seqs == y_seqs, ValueError('Number of X sequences does not match the number of y sequences')

print(f"X_toks: {X_toks}")
print(f"y_toks: {y_toks}")
print(f"X_seqs: {X_seqs}")
print(f"y_seqs: {y_seqs}")

print(f"Getting Val Stats")

X = torch.load(f'data/tensors/val/X/X_val.pt')
y = torch.load(f'data/tensors/val/y/Y_val.pt')

X_val_toks += X.shape[1] * X.shape[0]
y_val_toks += y.shape[1] * X.shape[0]
X_val_seqs += X.shape[0]
y_val_seqs += y.shape[0]

assert X_val_toks == y_val_toks, ValueError('Number of X tokens does not match the number of y tokens')
assert X_val_seqs == y_val_seqs, ValueError('Number of X sequences does not match the number of y sequences')

print(f"X_val_toks: {X_val_toks}")
print(f"y_val_toks: {y_val_toks}")
print(f"X_val_seqs: {X_val_seqs}")
print(f"y_val_seqs: {y_val_seqs}")