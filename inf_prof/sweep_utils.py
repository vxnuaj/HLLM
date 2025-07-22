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

from transformers import PreTrainedTokenizerFast, AutoTokenizer
from huggingface_hub import login as hf_login
from typing import Dict, Optional
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

load_dotenv()
hf_login(os.getenv("HF_TOKEN"))

def load_config(config_path, id = None) -> Dict:
    assert id is not None, ValueError("id must not be None")
    if config_path.endswith("json"):
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_path.endswith("yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    logger.info(f"Loading {id} config from {config_path}.")
    return config # -> dict

def load_tokenizer(tokenizer_path, **kwargs):
  
    '''
    Load a tokenizer from huggingface model hub or local file
    
    Parameters
    ----------
    tokenizer_path : str
        The path to the tokenizer file.
    **kwargs
        Additional keyword arguments to pass to the tokenizer's from_pretrained method.
    
    Returns
    -------
    tokenizer : transformers.PreTrainedTokenizerFast or transformers.AutoTokenizer
        The loaded tokenizer.
    '''
    
    if type == "PreTrainedTokenizerFast":
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, **kwargs)
    elif type == "AutoTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
    
    return tokenizer

def load_model(
    model_path:str = None, 
    model:Optional = None, 
    return_model_weights:bool = True, 
    load_from_hf:bool = True, 
    return_tokenizer:bool = False, 
    load_tokenizer_config:dict = None
    ):
  
    tokenizer_path = load_tokenizer_config.pop("tokenizer_path")
   
    if return_tokenizer:
        tokenizer = load_tokenizer(tokenizer_path, **load_tokenizer_config)
 
    if not load_from_hf: 
        logger.info(f"Loading model weights from {model_path} from local.")
        model_weights = torch.load(model_path, map_location=torch.device('cpu'), weights_only = True)
        model.load_state_dict(model_weights)
        
        if not return_model_weights:
            del model_weights
        
    else:
        logger.info(f"Loading model from {model_path} from huggingface.")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model_weights = model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only = True))
        
        if not return_model_weights:
            del model_weights

    if return_model_weights and return_tokenizer:
        return model, tokenizer, model_weights
    elif return_model_weights and not return_tokenizer:
        return model, model_weights
    elif not return_model_weights and return_tokenizer:
        return model, tokenizer
    else:
        return model
 
def create_combinations(sweep_config):
    '''
    create all possible combinations of the sweep config as a dictionary.
    
    if quantization is True, then mixed precision is False and vice versa. 
    '''

    logger.info(f"Computing the cartesian product over all possible base inference configurations.")
    
    mp_dtype_count = len([dtype for dtype in sweep_config['mixed_precision_dtype'] if dtype is not None])
    q_dtype_count = len([dtype for dtype in sweep_config['quantization_wbits'] if dtype is not None])
    q_method_count = len([method for method in sweep_config['quantization_method'] if method is not None])
    kvcache_count = len(sweep_config['quantize_kvcache'])
    backend_count = len(sweep_config['model_execution_backend'])
    decoding_count = len(sweep_config['decoding'])
    
    mixed_precision_base_count = mp_dtype_count * kvcache_count * backend_count * decoding_count 
    quantization_base_count = q_dtype_count * q_method_count * kvcache_count * backend_count * decoding_count 
    neither_base_count = kvcache_count * backend_count * decoding_count 
    
    total_base_combinations = mixed_precision_base_count + quantization_base_count + neither_base_count
    
    torch_compile_backend_count = len(sweep_config['torch_compile_backend'])
    torch_compile_mode_count = len(sweep_config['torch_compile_mode'])
    
    torch_compile_combinations = ( # not multiplying by model_execution_backend because for torch.compile it is fixed to torch.compile (or * 1)
        mp_dtype_count * kvcache_count * decoding_count +  
        q_dtype_count * q_method_count * kvcache_count * decoding_count +  
        kvcache_count * decoding_count 
    )
    
    torch_compile_expanded_count = torch_compile_combinations * torch_compile_backend_count * torch_compile_mode_count
    non_torch_compile_count = total_base_combinations - torch_compile_combinations
    total_final_combinations = torch_compile_expanded_count + non_torch_compile_count
    
    logger.info(f"Base combinations: {total_base_combinations}")
    logger.info(f"Torch.compile base combinations: {torch_compile_combinations}")
    logger.info(f"Torch.compile expanded combinations: {torch_compile_expanded_count}")
    logger.info(f"Non-torch.compile combinations: {non_torch_compile_count}")
    logger.info(f"Total final combinations: {total_final_combinations}")
    
    final_combinations = []
    
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
        [dtype for dtype in sweep_config['quantization_wbits'] if dtype is not None],  
        [method for method in sweep_config['quantization_method'] if method is not None],  
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
    
    all_base_combinations = itertools.chain(
        mixed_precision_combinations,
        quantization_combinations, 
        neither_combinations
    )
   
    assert len(list(mixed_precision_combinations)) == mixed_precision_base_count, ValueError("Mixed precision base count does not match.")
    assert len(list(quantization_combinations)) == quantization_base_count, ValueError("Quantization base count does not match.")
    assert len(list(neither_combinations)) == neither_base_count, ValueError("Neither base count does not match.")
    assert len(list(all_base_combinations)) == total_base_combinations, ValueError("Total base count does not match.")
    
    with tqdm(total=total_final_combinations, desc="Generating combinations") as pbar:
        for combo in all_base_combinations:
            mixed_precision_dtype, quantization_wbits, quantization_method, quantize_kvcache, model_execution_backend, decoding = combo
            
            base_config = {
                'mixed_precision_dtype': mixed_precision_dtype,
                'quantization_wbits': quantization_wbits,
                'quantization_method': quantization_method,
                'quantize_kvcache': quantize_kvcache,
                'model_execution_backend': model_execution_backend,
                'decoding': decoding
            }
               
            for torch_backend in sweep_config['torch_compile_backend']:
                for torch_compile_mode in sweep_config['torch_compile_model']:
                    config = base_config.copy()
                    config['torch_compile_backend'] = torch_backend
                    config['torch_compile_mode'] = torch_compile_mode
                    final_combinations.append(config)
                    pbar.update(1)
                
            '''           
            
            # NOTE | this should be used instead if we end up adding extra model_execution_backends other than torch.compile
            
            if model_execution_backend == "torch.compile":
                for torch_backend in sweep_config['torch_compile_backend']:
                    for torch_mode in sweep_config['torch_compile_mode']:
                        config = base_config.copy()
                        config['torch_compile_backend'] = torch_backend
                        config['torch_compile_mode'] = torch_mode
                        final_combinations.append(config)
                        pbar.update(1)
            else:
                config = base_config.copy()
                config['torch_compile_backend'] = None
                config['torch_compile_mode'] = None
                final_combinations.append(config)
                pbar.update(1)
            '''
    
    return final_combinations

def compile_model(model, backend, mode):
    model = torch.compile(model, backend = backend, mode = mode)
    return model 
 
def quantize_kv():
    return
  
def main(
    model_path:str = None,
    tokenizer_path:str = None,
    sweep_config_path:str = None,
    q_model_path_table:dict = None,
    ):

    '''

    weights_path: path to the weights file.
    sweep_config_path: path to the sweep config file.
    model_config_path: path to the model config file.
    q_weights_path_table: dictionary with the quantized weights paths.

        {
            "gptq": "path/to/quantized/weights" or "path/to/hf/model",
            "awq": "path/to/quantized/weights" or "path/to/hf/model",
            "gguf": "path/to/quantized/weights" or "path/to/hf/model",
            "exl12": "path/to/quantized/weights" or "path/to/hf/model",
            "bitsandbytes": "path/to/quantized/weights" or "path/to/hf/model",
            "none": "path/to/non_quantized/weights" or "path/to/hf/model"
        }

    TODO
   
    - [ ] implement decoding methods
        - [ ] implement speculative decoding impl.
        - [ ] implement speculative top p decoding impl.
        - [ ] implement speculative top k decoding impl.
        - [ ] implement base top p decoding impl.
        - [ ] implement base top k decoding impl.
    - [ ] implement quantizing kv cache
    - [ ] implement mixed precision
    
    '''

    sweep_config = load_config(sweep_config_path, id = "sweep") 
    logger.info(f"Initializting model and loading weights.")

    configs = create_combinations(sweep_config)
  
    configs = tqdm(configs, desc = "Sweeping Hyperparameters", total = len(configs))
   
    for cfg in configs:
        # cfg is type dict, as 
        # {'mixed_precision_dtype': 'fp16', 'quantization_wbits': 'int8', 'quantization_method': 'gptq', 'quantize_kvcache': 'float16', \
        # 'model_execution_backend': 'torch.compile', 'decoding': 'speculative', 'torch_compile_backend': 'inductor', 'torch_compile_mode': 'default'}
      
        configs.set_description(f"Sweeping Hyperparameters | Mixed Precision: {cfg['mixed_precision_dtype']} | Quantization: {cfg['quantization_method']} \
                                | Quantization Wbits: {cfg['quantization_wbits']} | Quantization Kvcache: {cfg['quantize_kvcache']} | \
                                Model Execution Backend: {cfg['model_execution_backend']} | Decoding: {cfg['decoding']} | Torch Compile Backend: \
                                {cfg['torch_compile_backend']} | Torch Compile Mode: {cfg['torch_compile_mode']}")
        
        quantized_model_path = q_model_path_table[cfg['quantization_method']]

        model, tokenizer = load_model(
            model_path = quantized_model_path, 
            model = model, 
            return_model_weights = False, 
            load_from_hf = True,
            return_tokenizer = True,
            load_tokenizer_config = {'load_tokenzier_path': tokenizer_path}
            ) 
    
        if cfg['model_execution_backend'] == "torch.compile":
            model = torch.compile(model = model, backend = cfg['torch_compile_backend'], mode = cfg['torch_compile_mode'])
       
        # TODO - left off here. 
         
          
    return

if __name__ == "__main__":
    main(sweep_config_path = "inf_prof/search_space.yaml", debug_run = True)