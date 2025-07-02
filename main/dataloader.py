import torch
import os
import math
import random
import numpy

from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler

class TinyStoriesDataset(Dataset):
    def __init__(self, X, y, context_length=512):
        """
        Initialize the dataset with in-memory tensors.

        Args:
            X: Tensor containing the input data
            y: Tensor containing the corresponding labels
            context_length: Length of each sequence in the batch
        """
        assert len(X) == len(y), "X and y must have the same length"
        assert len(X.shape) == 2, f"Expected 2D tensor (batch, sequence), got {X.shape}"
        assert X.shape[1] == context_length, f"Sequence length {X.shape[1]} doesn't match context_length {context_length}"
        
        self.X = X
        self.y = y
        self.total_len = len(X)
        
        print(f'Dataset initialized with {self.total_len} samples, sequence length: {context_length}')

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.total_len

    def __getitem__(self, idx):
        """
        Fetch an item by index.

        Returns:
            Tuple of (X[idx], y[idx]) where each is a sequence of length context_length
        """
        return self.X[idx], self.y[idx]

def get_data(path, context_length=512):
    """
    Load all data from the specified directory into memory as tensors.

    Args:
        path: Directory containing 'X' and 'y' subdirectories with data files
        context_length: Expected length of each sequence

    Returns:
        Tuple of (X, y), where X and y are concatenated tensors from all files

    Raises:
        OSError: If no data files are found in the directories
        ValueError: If the total number of samples in X and y do not match or shapes are incorrect
    """
    X_dir = os.path.join(path, 'X')
    y_dir = os.path.join(path, 'y')

    if not os.path.exists(X_dir) or not os.path.exists(y_dir):
        raise OSError(f"X or y directory not found in {path}")

    X_pths = sorted([os.path.join(X_dir, f) for f in os.listdir(X_dir) if f.endswith('.pt')])
    y_pths = sorted([os.path.join(y_dir, f) for f in os.listdir(y_dir) if f.endswith('.pt')])

    if not X_pths or not y_pths:
        raise OSError(f"No .pt files found in {X_dir} or {y_dir}")

    print(f'Loading data from {len(X_pths)} X files and {len(y_pths)} y files')
    
    X_list = []
    y_list = []
    
    for x_path, y_path in zip(X_pths, y_pths):
        try:
            x = torch.load(x_path, map_location='cpu')
            y = torch.load(y_path, map_location='cpu')
            
            if len(x.shape) == 1:
                x = x.unsqueeze(0)  
            if len(y.shape) == 1:
                y = y.unsqueeze(0)
                
            seq_len = min(x.size(1), context_length)
            x = x[:, :seq_len]
            y = y[:, :seq_len]
            
            X_list.append(x)
            y_list.append(y)
            
        except Exception as e:
            print(f"Error loading {x_path} or {y_path}: {str(e)}")
            continue
    
    if not X_list or not y_list:
        raise ValueError("No valid data files could be loaded")
    
    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)
    
    if X.size(0) != y.size(0):
        raise ValueError(f"Mismatched number of samples: X has {X.size(0)}, y has {y.size(0)}")
    
    print(f'Successfully loaded {X.size(0)} samples with sequence length {X.size(1)}')
    return X, y

def get_dataloader(
    X,
    y,
    batch_size,
    num_workers,
    shuffle: bool = False,
    pin_memory: bool = False,
    parallelism_type: str = None,
    rank: int = None,
    *args,
    **kwargs
    ):
    
    """
    Create a DataLoader for the dataset, with support for distributed training.

    Args:
        X: Tensor containing the input data
        y: Tensor containing the corresponding labels
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        shuffle: Whether to shuffle the data (handled by sampler in distributed mode)
        pin_memory: Whether to pin memory for faster GPU transfer
        parallelism_type: 'fsdp', 'ddp', 'dp', or None (determines sampler type)
        rank: Process rank (required for FSDP/DDP)

    Returns:
        DataLoader configured for the specified parallelism type
    """
    
    dataset = TinyStoriesDataset(X, y)
    _val_distributed = ['fsdp', 'ddp']

    if parallelism_type in _val_distributed:
        if rank is None:
            raise ValueError("Rank must be provided for DDP/FSDP")

        sampler = DistributedSampler(
            dataset,
            rank=rank,
            shuffle=shuffle
        )
        
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader