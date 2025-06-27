import os
import torch
import json

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union

@dataclass
class TrainingConfig:
    vocab_size: int
    context_length: int
    epochs: int
    checkpoint_steps: Optional[int] 
    save_checkpoint_path: str
    save_hf: bool
    val_steps: str
    mixed_precision: bool
    max_grad_norm: float
    track_grad_norm: bool
    parallel_type: str
    val_mixed_precision: bool
    val_mixed_precision_dtype: torch.dtype
    fsdp_wrap_policy: str
    wandb: bool
    log_level: str
    _compile: bool
    _compile_warmup_steps: int
    hf_repo_config: Dict[str, Any] = field(default_factory=dict)
    mixed_precision_dtype: torch.dtype = torch.float16
    disable: Union[Optional[bool], Optional[int]] = None
    disable_exclude: Union[Optional[bool], Optional[int]] = None
    extra_args: Dict[str, Any] = field(default_factory=dict)

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
    model_name:str
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
    assert config_type in ['criterion', 'lr', 'opt', 'train', 'wandb', 'model', 'dataloader'], ValueError("config_type must be in 'loss', 'lr', 'opt' or 'train'")
    if config_type == 'criterion':
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
        with open(os.path.join(root_path, 'dataloader_config.json'), 'r') as f:
            return json.load(f)