import os
import torch
import json

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union

@dataclass
class TrainingConfig:
    """Configuration for the training process.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        context_length (int): Maximum context length for the model.
        epochs (int): Number of training epochs.
        checkpoint_steps (Optional[int]): Number of steps between saving checkpoints.
        save_checkpoint_path (str): Path to save model checkpoints.
        save_hf (bool): Whether to save the model in Hugging Face format.
        val_steps (str): Validation frequency (e.g., '100steps', '1epoch').
        mixed_precision (bool): Whether to use mixed precision training.
        max_grad_norm (float): Maximum gradient norm for clipping.
        track_grad_norm (bool): Whether to track gradient norm.
        parallel_type (str): Type of distributed training (e.g., 'DDP', 'FSDP').
        val_mixed_precision (bool): Whether to use mixed precision during validation.
        val_mixed_precision_dtype (torch.dtype): Data type for mixed precision validation.
        fsdp_wrap_policy (str): FSDP wrapping policy.
        wandb (bool): Whether to use Weights & Biases for logging.
        log_level (str): Logging level (e.g., 'INFO', 'DEBUG').
        _compile (bool): Whether to compile the model using `torch.compile`.
        _compile_warmup_steps (int): Number of warmup steps for `torch.compile`.
        hf_repo_config (Dict[str, Any]): Configuration for Hugging Face repository.
        mixed_precision_dtype (torch.dtype): Data type for mixed precision training.
        log_root_path (Optional[str]): Root path for logging.
        load_checkpoint (bool): Whether to load a checkpoint.
        load_checkpoint_path (Optional[str]): Path to the checkpoint to load.
        save_on_interrupt (bool): Whether to save on interrupt signal.
        extra_args (Dict[str, Any]): Additional arguments.
    """
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
    log_root_path: Optional[str] = None
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None
    save_on_interrupt: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        """Initializes the TrainingConfig.

        Args:
            **kwargs: Keyword arguments to initialize the dataclass fields.
        """
        fields = TrainingConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class SchedulerConfig:
    """Configuration for the learning rate scheduler.

    Attributes:
        warmup_steps (int): Number of warmup steps.
        constant_steps (int): Number of constant learning rate steps.
        decay_steps (int): Number of decay steps.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        extra_args (dict): Additional arguments for the scheduler.
    """
    warmup_steps:int
    constant_steps:int
    decay_steps:int
    max_lr:float
    min_lr:float
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        """Initializes the SchedulerConfig.

        Args:
            **kwargs: Keyword arguments to initialize the dataclass fields.
        """
        fields = SchedulerConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class OptimizerConfig:
    """Configuration for the optimizer.

    Attributes:
        lr (float): Learning rate.
        betas (list): Coefficients used for computing running averages of gradient and its square.
        eps (float): Term added to the denominator to improve numerical stability.
        weight_decay (float): Weight decay (L2 penalty).
        fused (bool): Whether to use fused optimizer (if available).
        extra_args (dict): Additional arguments for the optimizer.
    """
    lr:float
    betas:list
    eps:float
    weight_decay:float
    fused:bool
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        """Initializes the OptimizerConfig.

        Args:
            **kwargs: Keyword arguments to initialize the dataclass fields.
        """
        fields = OptimizerConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class CriterionConfig:
    """Configuration for the loss criterion.

    Attributes:
        reduction (str): Specifies the reduction to apply to the output (e.g., 'mean', 'sum').
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the input gradient.
        extra_args (dict): Additional arguments for the criterion.
    """
    reduction:str
    ignore_index:int
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        """Initializes the CriterionConfig.

        Args:
            **kwargs: Keyword arguments to initialize the dataclass fields.
        """
        fields = CriterionConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging.

    Attributes:
        project (str): Name of the Weights & Biases project.
        name (str): Name of the current run.
        entity (str): Weights & Biases entity (username or team name).
        tags (list): List of tags for the run.
        notes (str): Notes for the run.
        id (str): Unique ID for the run.
        extra_args (dict): Additional arguments for Weights & Biases.
    """
    project:str
    name:str
    entity:str
    tags:list
    notes:str
    id: str
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        """Initializes the WandbConfig.

        Args:
            **kwargs: Keyword arguments to initialize the dataclass fields.
        """
        fields = WandbConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs

@dataclass
class ModelConfig:
    """Configuration for the model architecture.

    Attributes:
        context_len (int): Context length of the model.
        d_model (int): Dimension of the model embeddings.
        n_heads (int): Number of attention heads.
        n_blocks (int): Number of transformer blocks.
        vocab_size (int): Size of the vocabulary.
        pos_emb_dropout_p (float): Dropout probability for positional embeddings.
        pos_emb_type (str): Type of positional embedding (e.g., 'rope', 'learned').
        learned (bool): Whether positional embeddings are learned.
        ntk_rope_scaling (bool): Whether to apply NTK RoPE scaling.
        dyn_scaling (bool): Whether to apply dynamic scaling.
        attn_type (str): Type of attention mechanism (e.g., 'flash', 'standard').
        n_groups (int): Number of groups for grouped query attention.
        top_k_sparsev (int): Top-k for sparse attention.
        p_threshold (int): Threshold for pruning attention probabilities.
        p_threshold_steps_fraction (float): Fraction of steps over which to apply pruning threshold.
        flash_attn (bool): Whether to use Flash Attention.
        flash_attn_dtype (torch.dtype): Data type for Flash Attention.
        supress_warnings (bool): Whether to suppress warnings.
        verbose (bool): Whether to enable verbose logging.
        model_name (str): Name of the model.
        model_series_name (str): Name of the model series.
        extra_args (dict): Additional arguments for the model.
    """
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
    model_series_name:str
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        """Initializes the ModelConfig.

        Args:
            **kwargs: Keyword arguments to initialize the dataclass fields.
        """
        fields = ModelConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs 
    
@dataclass
class DataloaderConfig:
    """Configuration for data loaders.

    Attributes:
        train_dataloader_config (dict): Configuration for the training data loader.
        val_dataloader_config (dict): Configuration for the validation data loader.
        extra_args (dict): Additional arguments for the data loader.
    """
    train_dataloader_config:dict
    val_dataloader_config:dict 
    extra_args: dict = field(default_factory = dict)

    def __init__(self, **kwargs):
        """Initializes the DataloaderConfig.

        Args:
            **kwargs: Keyword arguments to initialize the dataclass fields.
        """
        fields = DataloaderConfig.__dataclass_fields__
        for key in fields:
            setattr(self, key, kwargs.pop(key, None))
        self.extra_args = kwargs    

def get_config(root_path:str, config_type:str):
    """Loads configuration from a JSON file.

    Args:
        root_path (str): The root directory where configuration files are located.
        config_type (str): The type of configuration to load (e.g., 'criterion', 'lr', 'opt', 'train', 'wandb', 'model', 'dataloader').

    Returns:
        dict: The loaded configuration as a dictionary.

    Raises:
        ValueError: If an invalid `config_type` is provided.
    """
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