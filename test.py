import torch
from tqdm import tqdm
import os

print(f"Number of X tensors: {len(os.listdir('data/tensors/train/X'))}")
print(f"Number of y tensors: {len(os.listdir('data/tensors/train/y'))}")

X_samples = 0
y_samples = 0

for i in tqdm(range(len(os.listdir('data/tensors/train/X')))):
    X = torch.load(f'data/tensors/train/X/X_train_{i}.pt')
    y = torch.load(f'data/tensors/train/y/y_train_{i}.pt')


    X_samples += X.shape[0]
    y_samples += y.shape[0]

    del X, y
    
assert X_samples == y_samples, ValueError("Number of X samples does not match number of y samples")

print(X_samples)
print(y_samples)