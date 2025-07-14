'''
# TODO 

- [ ] define the full search space
- [ ] write functions
    - [ ] quantization and mixed precision must not co exist.
    - [X] compile
    - [ ] quantize
    - [ ] quantize_kvcache
- [ ] enable to test benchmarks for a given config
- [ ] enable to test inference speed for a given config
'''

import os
import sys
import yaml
import torch
import json
import itertools
import logging

from typing import Union
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import Athena
from quantize import quantize_gptq

logger = logging.getLogger(__name__)

def load_config(config_path, id = None):
    assert id is not None, ValueError("id must not be None")
    if config_path.endswith("json"):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith("yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    logger.info(f"Loading {id} config from {config_path}.")
    return config

def load_weights_model(model, weights_path, load_model_weights = True):
    model_weights = torch.load(weights_path, weights_only = True )
    if load_model_weights:
        model.load_state_dict(model_weights)
        return model, model_weights
 
def create_combinations(sweep_config):
    '''
    create all possible combinations of the sweep config as a dictionary.
    
    if quantization is True, then mixed precision is False and vice versa. 
    '''

    logger.info(f"Computing the cartesian product over all possible base inference configurations.")
    
    mixed_precision_combinations = itertools.product(
        [dtype for dtype in sweep_config['mixed_precision_dtype'] if dtype is not None], 
        [None],
        [None],
        sweep_config['quantize_kvcache'],
        sweep_config['model_execution_backend'],
        sweep_config['decoding']
    )
    
    quantization_combinations = itertools.product(
        [None], 
        [dtype for dtype in sweep_config['quantization_dtype'] if dtype is not None],  
        sweep_config['quantization_method'],
        sweep_config['quantize_kvcache'],
        sweep_config['model_execution_backend'],
        sweep_config['decoding']
    )
    
    neither_combinations = itertools.product(
        [None], 
        [None], 
        [None], 
        sweep_config['quantize_kvcache'],
        sweep_config['model_execution_backend'],
        sweep_config['decoding']
    )
    
    base_combinations = itertools.chain(mixed_precision_combinations, quantization_combinations, neither_combinations)
  
    all_configs = []
    torch_compile_combinations = itertools.product(sweep_config['torch_compile_backend'], sweep_config['torch_compile_mode'])
    
    mp_count = len([d for d in sweep_config['mixed_precision_dtype'] if d is not None])
    q_count = len([d for d in sweep_config['quantization_dtype'] if d is not None])
    base_count = (mp_count + q_count + 1) * len(sweep_config['quantize_kvcache']) * len(sweep_config['model_execution_backend']) * len(sweep_config['decoding'])
    total_count = base_count * len(list(torch_compile_combinations))
    
    base_combinations = tqdm(base_combinations, desc="Computing the Cartesian Product of Hyperparameter Configurations.", total=total_count)
    
    for mixed_precision_dtype, quantization_dtype, quantization_method, quantize_kvcache, model_execution_backend, decoding in base_combinations:
        
        base_config = {
            "mixed_precision_dtype": mixed_precision_dtype,
            "quantization_dtype": quantization_dtype,
            "quantization_method": quantization_method,
            "quantize_kvcache": quantize_kvcache,
            "model_execution_backend": model_execution_backend,
            "decoding": decoding
            }

        for torch_compile_backend, torch_compile_mode in torch_compile_combinations:
            config = base_config.copy()
            config['torch_compile_backend'] = torch_compile_backend
            config['torch_compile_mode'] = torch_compile_mode
            all_configs.append(config)
            base_combinations.update(1)
            
    '''
    mixed_precision_dtype: ["fp16", "bfloat16", "fp8", "bfloat8", "int8", "int4", None]
    quantization_dtype: ["float16", "float8", "bfloat16", "bfloat8", "int8", "int4", None]
    quantization_method: ["gptq", "awq", "gguf", "exl12", "bitsandbytes"]
    quantize_kvcache: ["float16", "bfloat16", "float8", "bfloat8", "int8", "int4", None]
    model_execution_backend: ["torch.compile", "onnxruntime", "tensorrt", "vllm"]
    torch_compile_backend: ["inductor", "eager", "aot_eager"]
    torch_compile_mode: ["default", "reduce-overhead", "max-autotune"]
    decoding: ["speculative",  "speculative_top_p",  "speculative_top_k", "base_top_p",  "base_top_k"] 
    '''

    
    ''' 
    for mixed_precision, quantization, model_execution_backend, decoding, quantize_kvcache in base_combinations:
        mp_dtypes = sweep_config["mixed_precision_dtype"]
        q_dtypes = sweep_config["quantization_dtype"]
        q_methods = sweep_config["quantization_method"]
        torch_backends = sweep_config["torch_compile_backend"]
        torch_compile_modes = sweep_config["torch_compile_mode"] 
       
        for mp_dtype, q_dtype, q_method, torch_backend, torch_compile_mode in itertools.product(mp_dtypes, q_dtypes, q_methods, torch_backends, torch_compile_modes):
            config = {
                "mixed_precision": mixed_precision,
                "quantization": quantization,
                "model_execution_backend": model_execution_backend,
                "decoding": decoding,
                "quantize_kvcache": quantize_kvcache,
                }

            if config['mixed_precision']:
                config["mixed_precision_dtype"] = mp_dtype
                if config['quantization']:
                    config['quantization'] = False
            
            if config['quantization']:
                config["quantization_dtype"] = q_dtype
                config["quantization_method"] = q_method
                
                if config["mixed_precision"]:
                    config["mixed_precision"] = False
                
            if config['model_execution_backend'] == "torch.compile":
                config["torch_compile_backend"] = torch_backend
                config["torch_compile_mode"] = torch_compile_mode

            all_configs.append(config)

    '''

    logger.info(f"Computed and cached {len(all_configs)} configurations.")
    return all_configs
 
def _compile_model(model, backend, mode):
    model = torch.compile(model, backend = backend, mode = mode)
    return model 
 
def quantize(
    model, 
    quantization_method, 
    quantization_dtype:Union[str, list[str]], 
    calibration_data,
    groupsize = 128,
    percdamp = 0.1,
    nsamples = 128,
    ):
   
    '''
    TODO 
    - [ ] gptq
        - [ ] test if it works with model
        - [ ] cleanup code
    - [ ] awq
    - [ ] gguf
    - [ ] exl12
    - [ ] bitsandbytes
    '''
  
    if quantization_method == "gptq":
        return model
    elif quantization_method == "awq":
        pass
    elif quantization_method == "gguf":
        pass
    elif quantization_method == "exl12":
        pass
    elif quantization_method == "bitsandbytes":
        pass
   
    return

def quantize_kv():
    return
  
def main(
    model = None,
    weights_path:str = None,
    sweep_config_path:str = None,
    model_config_path:str = None,
    load_model_weights:bool = True,
    debug_run:bool = False
    ):

    '''
   
    TODO - remove the debug run for official run.
    
    '''

    sweep_config = load_config(sweep_config_path, id = "sweep") 

    if not debug_run: 
        model_config = load_config(model_config_path, id = "model")
    
    if not debug_run: 
        logger.info(f"Initializting model and loading weights.")
        model = Athena(**model_config)
        model, _ = load_weights_model(model, weights_path = weights_path, load_model_weights = load_model_weights)
   

    configs = create_combinations(sweep_config)
    
    return

if __name__ == "__main__":
    main(sweep_config_path = "inf_prof/search_space.yaml", debug_run = True)