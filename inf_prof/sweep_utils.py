'''
# TODO 

- [ ] deifine the full search space
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

from tqdm import tqdm
from pprint import pprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from model import LLaMA

def load_config(config_path, type = "json"):
    if type == "json":
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif type == "yaml":
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
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

    # note that when we print as `print(tuple(base_combinations))` the order of elements 
    # in the individual tuple is in the order as is ran in the function below
    
    base_combinations = itertools.product(
        sweep_config['mixed_precision'],
        sweep_config['quantization'],
        sweep_config['model_execution_backend'],
        sweep_config['decoding']
        )
   
    all_configs = []
       
    for mixed_precision, quantization, model_execution_backend, decoding in base_combinations:
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

    print(f"Generated {len(all_configs)} configurations" )
    return all_configs
 
def _compile_model(model, compile_backend, fullgraph):
    if compile_backend == "inductor":
        model = torch.compile(model, backend = "inductor")
    elif compile_backend == "eager":
        model = torch.compile(model, backend = "eager")
    elif compile_backend == "aot_eager":
        model = torch.compile(model, backend = "aot_eager")
    return model 
  
def main(
    model = None,
    weights_path:str = None,
    sweep_config_path:str = None,
    model_config_path:str = None,
    load_model_weights:bool = True,
    debug_run:bool = False
    ):

    logger = logging.getLogger(__name__)
    logger.info(f"Loading sweep & model_config from {sweep_config_path} & {model_config_path} respectively")
    
    sweep_config = load_config(sweep_config_path, type = "yaml")
    if not debug_run: 
        model_config = load_config(model_config_path, type = "json")
   
    logger.info(f"Initializting model and loading weights.")
    
    if not debug_run: 
        model = LLaMA(**model_config)
        model, _ = load_weights_model(model, weights_path = weights_path, load_model_weights = load_model_weights)
   
    logger.info(f"Computing the cartesian product over all possible inference configurations.")
    configs = create_combinations(sweep_config)
   
    pbar_configs_iter = tqdm(enumerate(configs), total = len(configs), desc = "Running Hyperparameter Sweep") 
    
    for i, cfg in pbar_configs_iter:
        pbar_configs_iter.set_description(f"Running Hyperparameter Sweep")
   
        if cfg['model_execution_backend'] == "torch.compile":
            model = torch.compile(model, backend = cfg['torch_compile_backend'])
            
    
    return

if __name__ == "__main__":
    main(sweep_config_path = "inf_prof/search_space.yaml", debug_run = True)