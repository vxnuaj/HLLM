import torch

X = torch.load('data/tensors/train/X/X_train_0.pt')
y = torch.load('data/tensors/train/y/y_train_0.pt')

num_samples = 10

for sample in range(num_samples):
    print(f"Sample {sample + 1}:")
    print(f"X: {X[sample, 0:10]}")
    print(f"y: {y[sample, 0:10]}")
    print()

