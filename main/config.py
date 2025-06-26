import os
import torch
import json

from dataclasses import dataclass, field

@dataclass 
class TrainingConfig:
    vocab_size:int
    batch_size:int
    context_length:int
    epochs:int
    checkpoint_steps:int
    save_checkpoint_path:str
    save_hf:bool
    hf_repo_config:dict
    val_steps:int
    mixed_precision:bool
    mixed_precision_dtype:torch.dtype
    max_grad_norm:float
    track_grad_norm:bool
    parallel_type:str
    val_batch_size:int
    val_num_workers:int
    val_shuffle:bool
    val_pin_memory:bool
    val_mixed_precision:bool
    X_val_path:str
    y_val_path:str
    wandb:bool
    _compile:bool
    _compile_warmup_steps:int
    log_level:str
    log_root_path:str
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = TrainingConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class SchedulerConfig:
    warmup_steps:int
    constant_steps:int
    decay_steps:int
    max_lr:float
    min_lr:float
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = SchedulerConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class OptimizerConfig:
    lr:float
    betas:list
    eps:float
    weight_decay:float
    fused:bool
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = OptimizerConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class CriterionConfig:
    reduction:str
    ignore_index:int
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = CriterionConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class WandbConfig:
    project:str
    name:str
    entity:str
    tags:list
    notes:str
    sweep_id:str
    sweep_config:str
    sweep_name:str
    sweep_entity:str
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = WandbConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class ModelConfig:
    context_len:int
    d_model:int
    n_heads:int
    n_blocks:int
    vocab_size:int
    pos_emb_dropout_p:float
    pos_emb_type:str
    learned:bool
    ntk_rope_scaling:bool
    dyn_scaling:bool
    attn_type:str
    n_groups:int
    top_k_sparsev:int
    p_threshold:int
    p_threshold_steps_fraction:float
    flash_attn:bool
    flash_attn_dtype:torch.dtype
    supress_warnings:bool
    verbose:bool
    _model_name:str
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = ModelConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs 
    
@dataclass
class DataloaderConfig:
    train_dataloader_config:dict
    val_dataloader_config:dict 
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        fields = DataloaderConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs    

def get_config(root_path:str, config_type:str):
    assert config_type in ['loss', 'lr', 'opt', 'train', 'wandb', 'model', 'dataloader'], ValueError("config_type must be in 'loss', 'lr', 'opt' or 'train'")
    if config_type == 'loss':
        with open(os.path.join(root_path, 'criterion_config.json'), 'r') as f:
            return json.load(f)
    elif config_type == 'lr':
        with open(os.path.join(root_path, 'lr_config.json'), 'r') as f:
            return json.load(f)           
    elif config_type == 'opt':
        with open(os.path.join(root_path, 'opt_config.json'), 'r') as f:
            return json.load(f)
    elif config_type == 'train':
        with open(os.path.join(root_path, 'train_config.json'), 'r') as f:
            return json.load(f)           
    elif config_type == 'wandb':
        with open(os.path.join(root_path, 'wandb_config.json'), 'r') as f:
            return json.load(f)           
    elif config_type == 'model':
        with open(os.path.join(root_path, 'model_config.json'), 'r') as f:
            return json.load(f)                  
    elif config_type == 'dataloader':
        with open(os.path.join(root_path, 'dataloader_config.json', 'r')) as f:
            return json.load(f)