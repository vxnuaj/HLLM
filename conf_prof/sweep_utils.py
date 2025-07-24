'''
Script to sweep across all possible configs for the model under `search_space.yaml`
'''

import os
import json
import shutil
import itertools
import functools
import warnings
import torch
import yaml
import numpy as np
import time
import sys
import torch.nn.functional as F
import torch.distributed as dist
import logging

from rich.progress import track
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import Adam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.blocks import TransformerBlock
from model.model import Athena

def load_config(path: str):
    """Loads a YAML configuration file and converts specified string-based torch.dtypes to actual torch.dtype objects.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        dict: The loaded configuration with torch.dtype strings converted.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    dtype_map = {
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16
    }
    
    if "flash_attn_dtype" in config:
        if isinstance(config["flash_attn_dtype"], list):
            config["flash_attn_dtype"] = [dtype_map[d] for d in config["flash_attn_dtype"]]
        else:
            config["flash_attn_dtype"] = [dtype_map[config["flash_attn_dtype"]]]
    return config

def generate_config_combinations(config):
    """Generates all valid combinations of configurations from a given dictionary.

    This function takes a configuration dictionary where values can be single items or lists.
    It expands lists into all possible combinations and applies specific filtering rules
    (e.g., setting 'n_groups' to None if 'attn_type' is not 'gqa').

    Args:
        config (dict): A dictionary where keys are configuration parameters and values are
                       either single values or lists of possible values.

    Returns:
        list: A list of dictionaries, where each dictionary represents a unique and valid
              combination of configuration parameters.
    """
    for k, v in config.items():
        if not isinstance(v, list):
            config[k] = [v]
    
    keys, values = zip(*config.items())
    valid_combos = []
    
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))

        if cfg['attn_type'] not in ['gqa']:
            cfg['n_groups'] = None
        
        if not cfg['flash_attn']:
            cfg['flash_attn_dtype'] = None
        
        valid_combos.append(cfg)
    
    return valid_combos

def prof_single_forward_pt(
    model, 
    data_shape, 
    vocab_size, 
    use_mixed_precision=False
    ):
    """Profiles a single forward pass of the model using PyTorch's profiler.

    Args:
        model (torch.nn.Module): The model to profile.
        data_shape (tuple): The shape of the input data (e.g., (batch_size, sequence_length)).
        vocab_size (int): The size of the vocabulary for generating random input data.
        use_mixed_precision (bool, optional): Whether to use mixed precision for the forward pass. Defaults to False.

    Returns:
        str: The path to the exported Chrome trace file.
    """
    input_data = torch.randint(0, vocab_size, size=data_shape, device=next(model.parameters()).device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        if use_mixed_precision:
            with autocast(device_type='cuda', dtype=torch.float16):
                model(input_data)
        else:
            model(input_data)
    return prof.export_chrome_trace("prof_single_forward_pt.json")

def time_avg_forward(
    model, 
    data_shape, 
    vocab_size, 
    n_inf_passes, 
    use_mixed_precision=False
    ):
    """Measures the average time for a single forward pass of the model.

    Args:
        model (torch.nn.Module): The model to measure.
        data_shape (tuple): The shape of the input data (e.g., (batch_size, sequence_length)).
        vocab_size (int): The size of the vocabulary for generating random input data.
        n_inf_passes (int): Number of inference passes to average over.
        use_mixed_precision (bool, optional): Whether to use mixed precision for the forward pass. Defaults to False.

    Returns:
        float: The average time taken for a forward pass in seconds.
    """
    
    device = next(model.parameters()).device
    input_data = torch.randint(0, vocab_size, size=data_shape, device=device)
    times = []

    for _ in range(n_inf_passes):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if use_mixed_precision:
            with autocast(device_type='cuda', dtype=torch.float16):
                start = time.perf_counter()
                model(input_data)
        else:
            start = time.perf_counter()
            model(input_data)
        elapsed = time.perf_counter() - start
        if elapsed >= 0 and not np.isnan(elapsed) and not np.isinf(elapsed):
            times.append(elapsed)
    return float(np.mean(times)) if times else 0.0

def time_avg_backward(
    model, 
    data_shape, 
    vocab_size, 
    n_bck_passes, 
    use_mixed_precision=False
    ):
    """Measures the average time for a single backward pass of the model.

    Args:
        model (torch.nn.Module): The model to measure.
        data_shape (tuple): The shape of the input data (e.g., (batch_size, sequence_length)).
        vocab_size (int): The size of the vocabulary for generating random input data.
        n_bck_passes (int): Number of backward passes to average over.
        use_mixed_precision (bool, optional): Whether to use mixed precision for the backward pass. Defaults to False.

    Returns:
        float: The average time taken for a backward pass in seconds.
    """
    
    device = next(model.parameters()).device
    input_data = torch.randint(0, vocab_size, size=data_shape, device=device)
    target = torch.randint_like(input_data, 0, vocab_size)
    times = []
    scaler = GradScaler(enabled=use_mixed_precision)

    for _ in range(n_bck_passes):
        model.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()  
        if use_mixed_precision:
            with autocast(device_type='cuda', dtype=torch.float16):
                start = time.perf_counter()
                out = model(input_data)
                loss = F.cross_entropy(out.view(-1, out.size(-1)), target.view(-1))
            scaler.scale(loss).backward()
        else:
            start = time.perf_counter()
            out = model(input_data)
            loss = F.cross_entropy(out.view(-1, out.size(-1)), target.view(-1))
            loss.backward()
        elapsed = time.perf_counter() - start
        if elapsed >= 0 and not np.isnan(elapsed) and not np.isinf(elapsed):
            times.append(elapsed)
    return float(np.mean(times)) if times else 0.0


def time_avg_fwd_backward(
    model, 
    data_shape, 
    vocab_size, 
    n_iter, 
    use_mixed_precision=False
    ):
    """Measures the average time for a combined forward and backward pass of the model.

    Args:
        model (torch.nn.Module): The model to measure.
        data_shape (tuple): The shape of the input data (e.g., (batch_size, sequence_length)).
        vocab_size (int): The size of the vocabulary for generating random input data.
        n_iter (int): Number of forward-backward passes to average over.
        use_mixed_precision (bool, optional): Whether to use mixed precision for the passes. Defaults to False.

    Returns:
        float: The average time taken for a forward-backward pass in seconds.
    """
    
    device = next(model.parameters()).device
    input_data = torch.randint(0, vocab_size, size=data_shape, device=device)
    target = torch.randint_like(input_data, 0, vocab_size)
    times = []
    scaler = GradScaler(enabled=use_mixed_precision)

    for _ in range(n_iter):
        model.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if use_mixed_precision:
            with autocast(device_type='cuda', dtype=torch.float16):
                start = time.perf_counter()
                out = model(input_data)
                loss = F.cross_entropy(out.view(-1, out.size(-1)), target.view(-1))
            scaler.scale(loss).backward()
        else:
            start = time.perf_counter()
            out = model(input_data)
            loss = F.cross_entropy(out.view(-1, out.size(-1)), target.view(-1))
            loss.backward()
        elapsed = time.perf_counter() - start
        if elapsed >= 0 and not np.isnan(elapsed) and not np.isinf(elapsed):
            times.append(elapsed)

    return float(np.mean(times)) if times else 0.0

def wrap_model(model, parallel_type, fsdp_wrap_policy="auto"):
    """Wraps the model with the specified parallelization strategy.

    Args:
        model (torch.nn.Module): The model to wrap.
        parallel_type (str): The type of parallelization to use ('none', 'ddp', or 'fsdp').
        fsdp_wrap_policy (str, optional): The FSDP wrapping policy ('auto' or 'transformer'). Defaults to "auto".

    Returns:
        torch.nn.Module: The wrapped model.

    Raises:
        ValueError: If an unknown `fsdp_wrap_policy` or `parallel_type` is provided.
    """
    if parallel_type == "none":
        return model
    if parallel_type == "ddp":
        return DDP(model)
    elif parallel_type == "fsdp":
        if fsdp_wrap_policy == "auto":
            return FSDP(model)
        elif fsdp_wrap_policy == "transformer":
            auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls = {TransformerBlock})
            return FSDP(model, auto_wrap_policy = auto_wrap_policy)
        else:
            raise ValueError(f"Unknown fsdp_wrap_policy: {fsdp_wrap_policy}")
    else:
        raise ValueError(f"Unknown parallel type: {parallel_type}")

def warmup_compile(model, compile_wamrup_steps, vocab_size, data_shape):
    """Performs warmup steps for a compiled model.

    Args:
        model (torch.nn.Module): The compiled model to warm up.
        compile_wamrup_steps (int): The number of warmup steps.
        vocab_size (int): The size of the vocabulary for generating random input data.
        data_shape (tuple): The shape of the input data (e.g., (batch_size, sequence_length)).
    """
    device = next(model.parameters()).device
    input_data = torch.randint(0, vocab_size, size = data_shape, device = device)
    for _ in range(compile_wamrup_steps):
        model(input_data)

def run_profs(
    cfg_path: str,
    data_shape: tuple,
    vocab_size: int = 10000,
    n_inf_passes: int = 50,
    n_bck_passes: int = 50,
    n_fwd_bck_iter: int = 50,
    results_root: str = "conf_prof/results/results_json",
    profile_forward: bool = False,
    compile_warmup_steps:int = 5,
    ):
    """Runs profiling for various model configurations and saves the results.

    This function iterates through different model configurations defined in a YAML file,
    initializes models, applies parallelization strategies, and measures forward, backward,
    and combined forward-backward pass times. It also supports profiling a single forward pass.

    Args:
        cfg_path (str): Path to the YAML configuration file defining the sweep space.
        data_shape (tuple): Shape of the input data for profiling (e.g., (batch_size, sequence_length)).
        vocab_size (int, optional): Vocabulary size for generating input data. Defaults to 10000.
        n_inf_passes (int, optional): Number of inference passes to average for timing. Defaults to 50.
        n_bck_passes (int, optional): Number of backward passes to average for timing. Defaults to 50.
        n_fwd_bck_iter (int, optional): Number of forward-backward iterations to average for timing. Defaults to 50.
        results_root (str, optional): Root directory to save profiling results. Defaults to "conf_prof/results/results_json".
        profile_forward (bool, optional): Whether to perform a detailed forward pass profile. Defaults to False.
        compile_warmup_steps (int, optional): Number of warmup steps for compiled models. Defaults to 5.
    """
   
    local_rank = int(os.environ['LOCAL_RANK']) 
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warnings.warn('Currently on CPU.')
        cont = input('Continue [y/n]?\n')
        if cont == 'n':
            sys.exit(0)

    raw = load_config(cfg_path)
    configs = generate_config_combinations(raw)

    grouped_configs = {"ddp": [], "fsdp": []}
    for cfg in configs:
        parallel_type = cfg.get("parallel", "none")
        grouped_configs[parallel_type].append(cfg)
   
    for parallel_type, config_group in grouped_configs.items():
        if torch.cuda.is_available() and dist.is_available():
            dist.init_process_group(backend='nccl')

        for i, cfg in enumerate(track(sequences = config_group, description = f"Sweeping Hyperparameters | Running {parallel_type.upper()} configs")):
            try:
                model = Athena(**cfg).to(device)
                model.train()

                use_mixed_precision = cfg.get("mixed_precision", False)
                use_compile = cfg.get("compile", False)

                if use_compile:
                    model = torch.compile(model)

                    warmup_compile(
                        model = model, 
                        compile_wamrup_steps = compile_warmup_steps, 
                        vocab_size = vocab_size, 
                        data_shape = data_shape
                        )

                model = wrap_model(model, parallel_type, fsdp_wrap_policy=cfg.get("fsdp_wrap_policy", "auto"))

                out_dir = os.path.join(results_root, f"{parallel_type}_config_{i}")
                os.makedirs(out_dir, exist_ok=True)

                if profile_forward:
                    prof_single_forward_pt(model, data_shape, vocab_size, use_mixed_precision)
                    shutil.move("prof_single_forward_pt.json", os.path.join(out_dir, "prof_pt.json"))

                avg_fwd = time_avg_forward(model, data_shape, vocab_size, n_inf_passes, use_mixed_precision)
                avg_bck = time_avg_backward(model, data_shape, vocab_size, n_bck_passes, use_mixed_precision)
                avg_fwdbck = time_avg_fwd_backward(model, data_shape, vocab_size, n_fwd_bck_iter, use_mixed_precision)

                cfg['flash_attn_dtype'] = str(cfg['flash_attn_dtype'])

                metrics = {
                    "config": cfg,
                    "avg_forward_time": avg_fwd,
                    "avg_backward_time": avg_bck,
                    "avg_fwd_bwd_time": avg_fwdbck
                }
                
                with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)

            except RuntimeError as e:
                logging.error(f"Error at iteration {i}: {e}")
                continue

        if dist.is_initialized():
            dist.destroy_process_group()

    print(f"Done! Results in '{results_root}'")