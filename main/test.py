import torch
import torch.distributed as dist
import torch.nn as nn
import argparse
import logging
import os
import time
import sys

from dataclasses import asdict
from tqdm import tqdm
from torch.amp import autocast
from dataloader import get_data, get_dataloader
from config import ModelConfig, CriterionConfig, DataloaderConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import Athena

def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(ch)
    
    return logger

class Tester:
    def __init__(
        self,
        model_config: dict,
        criterion_config: dict,
        dataloader_config: dict,
        mixed_precision: bool = False,
        mixed_precision_dtype: str = 'bfloat16',
        parallel_type: str = 'ddp',
    ):
        self.logger = setup_logger()
        self.mixed_precision = mixed_precision
        self.parallel_type = parallel_type
        
        # Set up distributed training if needed
        self._setup_parallel()
        self.device = self._get_device()
        
        # Initialize model
        self.model_config = ModelConfig(**model_config)
        self.model = self._get_model(asdict(self.model_config))
       
        criterion_config_dict = asdict(CriterionConfig(**criterion_config)) 
        del criterion_config_dict['extra_args']
        
        # Set up criterion
        self.criterion = nn.CrossEntropyLoss(**criterion_config_dict)
        
        # Set up dataloaders
        self.dataloader_config = DataloaderConfig(**dataloader_config)
        self.train_dataloader = self._get_dataloader('train')
        self.val_dataloader = self._get_dataloader('val')
        
        # Mixed precision
        self.mixed_precision_dtype = torch.bfloat16 if mixed_precision_dtype == 'bfloat16' else torch.float16
        
    def _setup_parallel(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            self.rank = rank
            self.world_size = world_size
            self.local_rank = local_rank
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
    
    def _get_device(self):
        return torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    def _get_model(self, model_config):
        model = Athena(**model_config)
        model = model.to(self.device)
        
        if self.parallel_type == 'ddp' and self.world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
        
        return model
    
    def _load_checkpoint(self):
        # Skip loading checkpoint for random initialization
        self.logger.info("Using randomly initialized model")
        return
    
    def _get_dataloader(self, split='train'):
        assert split in ['train', 'val'], "Split must be either 'train' or 'val'"
        
        config = self.dataloader_config.train_dataloader_config if split == 'train' else self.dataloader_config.val_dataloader_config
        
        # Get data
        X, y = get_data(config[f'{split}_data_root_path'])
        
        # Create dataloader
        dataloader = get_dataloader(
            X, 
            y,
            parallelism_type=self.parallel_type,
            rank=self.rank,
            batch_size=config[f'{split}_batch_size'],
            num_workers=config[f'{split}_num_workers'],
            shuffle=config[f'{split}_shuffle'],
            pin_memory=config[f'{split}_pin_memory']
        )
        
        return dataloader
    
    def evaluate(self, dataloader, split='val'):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            # Initialize tqdm progress bar
            progress_bar = tqdm(
                dataloader, 
                desc=f"Evaluating {split} | Loss: -",
                disable=(self.rank != 0)  # Only show for rank 0
            )
            
            for X, y in progress_bar:
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                
                if self.mixed_precision:
                    with autocast(device_type='cuda', dtype=self.mixed_precision_dtype):
                        logits = self.model(X)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                else:
                    logits = self.model(X)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # Calculate average loss for this batch
                batch_loss = loss.mean()
                
                # Update progress bar with current batch loss
                if self.rank == 0:  # Only update from rank 0
                    progress_bar.set_description(f"Evaluating {split} | Loss: {batch_loss.item():.4f}")
                
                # Gather loss across all processes
                if self.world_size > 1:
                    dist.all_reduce(batch_loss, op=dist.ReduceOp.SUM)
                    batch_loss = batch_loss / self.world_size
                
                total_loss += batch_loss.item() * X.size(0)
                total_samples += X.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return avg_loss
    
    def run(self, eval_train=False, eval_val=True):
        """
        Run evaluation on specified datasets.
        
        Args:
            eval_train (bool): If True, evaluate on training set
            eval_val (bool): If True, evaluate on validation set
            
        Returns:
            dict: Dictionary containing evaluation results
        """
        results = {}
        
        if eval_train:
            train_loss = self.evaluate(self.train_dataloader, 'train')
            self.logger.info(f"Training Loss: {train_loss:.4f}")
            results['train_loss'] = train_loss
        
        if eval_val:
            val_loss = self.evaluate(self.val_dataloader, 'val')
            self.logger.info(f"Validation Loss: {val_loss:.4f}")
            results['val_loss'] = val_loss
            
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on training and validation datasets')
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'], 
                        help='Mixed precision data type (default: bfloat16)')
    
    # Add evaluation mode arguments
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument('--train-only', action='store_true', help='Evaluate only on training set')
    eval_group.add_argument('--val-only', action='store_true', help='Evaluate only on validation set')
    # Default is to evaluate on both (handled in code)
    
    args = parser.parse_args()
    
    # Model configuration
    model_config = {
        "context_len": 512,
        "d_model": 512,
        "n_heads": 8,
        "n_blocks": 12,
        "vocab_size": 10000,
        "pos_emb_dropout_p": 0.1,
        "pos_emb_type": "rope",
        "learned": False,
        "ntk_rope_scaling": False,
        "dyn_scaling": False,
        "attn_type": "mhsa",
        "n_groups": None,
        "top_k_sparsev": None,
        "p_threshold": None,
        "p_threshold_steps_fraction": None,
        "flash_attn": True,
        "flash_attn_dtype": "float16",
        "supress_warnings": True,
        "model_name": "ATHENA_V1_TINY_39.7M",
        "model_series_name": "ATHENA \n 39.7M"
    }
    
    criterion_config = {
        "reduction": "none",
        "ignore_index": 0
    }

    dataloader_config = {
        "train_dataloader_config": {
            "train_batch_size": 64, 
            "train_num_workers": 12,
            "train_shuffle": True,
            "train_pin_memory": True,
            "train_data_root_path": "data/tensors/train"
        },
        "val_dataloader_config": {
            "val_batch_size": 64,
            "val_num_workers": 12,
            "val_shuffle": False,
            "val_pin_memory": True,
            "val_data_root_path": "data/tensors/val"
        }   
    }
    
    # Create tester instance with random initialization
    tester = Tester(
        model_config=model_config,
        criterion_config=criterion_config,
        dataloader_config=dataloader_config,
        mixed_precision=args.mixed_precision,
        mixed_precision_dtype=args.dtype
    )
    
    # Determine which datasets to evaluate on
    if args.train_only:
        eval_train, eval_val = True, False
    elif args.val_only:
        eval_train, eval_val = False, True
    else:  # default: evaluate both
        eval_train, eval_val = True, True
    
    # Run evaluation
    results = tester.run(eval_train=eval_train, eval_val=eval_val)