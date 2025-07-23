'''

TODO

- [ ] When we quantize, we should probably do a sweep over all set of quantized models to get the settings ( per quantization method ) with the best accuracy / inference speed.
    - wait but in that case we wouldn't need to run the quantization sweep over sweep_utils anymore... actually we would because we have different decoding methods...
    - so we should run the sweep and do best of n accuracy with ewach quantization method to extract best speed + accuracy
'''

import os
import sys
import json
import logging
import torch
import argparse

from transformers import (PreTrainedTokenizerFast, AutoModelForCausalLM, Tokenizer, \
    AwqConfig, QuantoConfig, VptqConfig, BitsAndBytesConfig, HqqConfig, SpQRConfig)
from typing import Union, Optional
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'package', 'gptq'))
from gptq import GPTQ
from quant import Quantizer
from model import Athena

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_tokenizer(
    tokenizer_file, 
    confirm_special_toks = True, 
    eos_token = "<|eos|>", 
    bos_token = "<|bos|>", 
    pad_token = "<|pad|>"
    ):
 
    '''
   
    tokenizer_file is a local .json file for the tokenizer. 
    
    '''
  
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = tokenizer_file)
    
    if confirm_special_toks:
        if tokenizer.eos_token != eos_token:
            tokenizer.eos_token = eos_token
        if tokenizer.bos_token != bos_token: 
            tokenizer.bos_token = bos_token
        if tokenizer.pad_token != pad_token:
            tokenizer.pad_token = pad_token
        
    return tokenizer
 
def quantize_gptq(
    calibration_data, 
    quantization_bits=4, 
    nsamples=128, 
    percdamp=0.01, 
    groupsize=128,
    tokenizer_file = None,
    hf_model_id = None,
    save_model_to_hf = False,
    save_model_hf_path:str = None,
    save_model_to_local = False,
    save_model_local_path:str = None,
    model_name:str = None
    ):

    assert save_model_path is not None, "Must specify a save model path ( or hf repo )"
   
    logger.info('Initializing GPTQ Quantization.')
       
    if isinstance(calibration_data, list, torch.tensor, torch.Tensor):
        calibration_data = tokenizer.batch_decode(calibration_data)
        if isinstance(calibration_data, list):
            calibration_data = torch.tensor(calibration_data)

    tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
    
    gptq_config = GPTQConfig(
        bits = quantization_bits,
        dataset = calibration_data,
        tokenizer=tokenizer,
        group_size = groupsize,
        batch_size = nsamples,
        damp_percent = percdamp,
    )
    
    quantized_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        device_map = 'auto',
        gptq_config=gptq_config,
    ) 
            
    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-gptq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-gptq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
        
def quantize_awq(
    hf_model_id:str,
    tokenizer:str,
    model_save_path: str,
    quantization_bits:int = 4,
    group_size:int = 128,
    do_fuse:bool = True,
    fuse_max_seq_len:int = 512,
    save_model_to_hf:bool = False,
    save_model_to_local:bool = False,
    save_model_hf_path:str = None,
    save_model_local_path:str = None,
    model_name:str = None
    ):

    assert model_save_path is not None, "Must specify a save model path ( or hf repo )"
    
    logger.info(f"Initializing AWQ Quantization")

    awq_config = AwqConfig(
        bits = quantization_bits,
        group_size = group_size,
        version = AWQLinerVersion.GEMv,
        do_fuse = do_fuse,
        fuse_max_seq_len = fuse_max_seq_len
    )
   
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model = hf_model_id,
        device_map = 'auto', 
        quantization_config = awq_config 
    )

    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-awq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-awq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))

    return quantized_model

def quantize_quanto(
    hf_model_id:str, 
    weights_target_dtype:str, 
    activations_target_dtype:str,
    save_model_to_hf:bool = False,
    save_model_to_local:bool = False,
    save_model_hf_path:str = None,
    save_model_local_path:str = None,
    model_name:str = None
    ):
    
    quanto_config = QuantoConfig(
        weights = weights_target_dtype,
        activations = activations_target_dtype
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        model = hf_model_id,
        device_map = 'auto',
        quantization_config = quanto_config
    )

    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-quanto" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-quanto" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))

    return quantized_model

def quantize_aqlm(
    hf_model_id:str,
    in_group_size,
    out_group_size,
    num_codebooks,
    nbits_per_codebook,
    save_model_to_hf:bool,
    save_model_to_local:bool,
    save_model_hf_path:str,
    save_model_local_path:str,
    model_name:str
    ):

    aqlm_config = AqlmConfig(
        in_group_size = in_group_size,
        out_group_size = out_group_size,
        num_codebooks = num_codebooks,
        num_codebooks = num_codebooks,
        nbits_per_codebook = nbits_per_codebook
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            quantized_config = aqlm_config
    )

    
    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-aqlm" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-aqlm" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))

    return quantized_model

def quantize_vptq(
    hf_model_id:str,
    config_for_layers, 
    shared_layer_config,
    save_model_to_hf:bool,
    save_model_to_local:bool,
    save_model_hf_path:str,
    save_model_local_path:str,
    model_name:str
    ):

    vptq_config = VptqConfig(
        config_for_layers = config_for_layers,
        shared_layer_config = shared_layer_config     
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantized_config = vptq_config
    )


    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-vptq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-vptq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))

    return quantized_model

def quantize_hqq(
    hf_model_id:str,
    nbits:int,
    group_size:int,
    save_model_to_hf:bool,
    save_model_to_local: bool,
    save_model_hf_path:str,
    save_model_local_path:str,
    model_name:str,
    skip_modules:List[str] = []
    ):


    hqq_config = HqqConfig(
        nbits = nbits,
        group_size = group_size,
        skip_modules = skip_modules
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantized_config = hqq_config
    )

    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-hqq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-hqq" if "/" not in save_model_hf_path else save_model_hf_path
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    return quantized_model

def quantize_bitsandbytes(
    hf_model_id:str,
    load_in_8bit:bool,
    load_in_4bit:bool,
    load_in_2bit:bool,
    load_in_1bit:bool,
    save_model_to_hf:bool,
    save_model_to_local:bool,
    save_model_path:str,
    model_name:str,
    ):

    _val_list = [load_in_8bit, load_in_4bit, load_in_2bit, load_in_1bit]

    assert _val_list.count(True) == 1, "Must specify exactly one of load_in_8bit, load_in_4bit, load_in_2bit, or load_in_1bit"

    bitsandbytes_config = BitsAndBytesConfig(
        load_in_8bit = load_in_8bit,
        load_in_4bit = load_in_4bit,
        load_in_2bit = load_in_2bit,
        load_in_1bit = load_in_1bit,
    )
    
    quantized_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantized_config = bitsandbytes_config
    )

    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-bitsandbytes" if "/" not in save_model_hf_path else save_model_hf_path
        logger.info(f"Saving quantized model to Hugging Face Hub: {repo_id}")
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-bitsandbytes" if "/" not in save_model_hf_path else save_model_hf_path
        logger.info(f"Saving quantized model to Hugging Face Hub and local: {repo_id}")
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        logger.info(f"Saving quantized model to local: {save_model_local_path}/{model_name}")
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    return quantized_model
        
def quantize_spqr(
    hf_model_id:str,
    bits:int,
    beta1:int,
    beta2:int,
    shapes:List[int],
    modules_not_to_convert:List[str],
    save_model_to_hf:bool,
    save_model_hf_path:str,
    save_model_to_local:bool,
    save_model_local_path:str,
    model_name:str
    ):

    spqr_config = SpQRConfig(
        bits = bits,
        beta1 = beta1,
        beta2 = beta2,
        shapes = shapes,
        modules_not_to_convert = modules_not_to_convert,
    )

    quantized_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        quantized_config = spqr_config
    )

    if save_model_to_hf and not save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-spqr" if "/" not in save_model_hf_path else save_model_hf_path
        logger.info(f"Saving quantized model to Hugging Face Hub: {repo_id}")
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )
    elif save_model_to_hf and save_model_to_local:
        repo_id = f"{save_model_hf_path}/{model_name}-spqr" if "/" not in save_model_hf_path else save_model_hf_path
        logger.info(f"Saving quantized model to Hugging Face Hub and local: {repo_id}")
        quantized_model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        tokenizer = load_tokenizer(tokenizer_file, eos_token = "<|eos|>", bos_token = "<|bos|>", pad_token = "<|pad|>")
        tokenizer.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload tokenizer for {model_name}",
            private=False
        )   
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    elif save_model_to_local and not save_model_to_hf:
        logger.info(f"Saving quantized model to local: {save_model_local_path}/{model_name}")
        torch.save(quantized_model.state_dict(), os.path.join(save_model_local_path, model_name))
    return quantized_model

def quantize(
    model,
    quantization_method,
    quant_save_path:str,
    model_id:str,
    gptq_config = None,
    awq_config = None,
    bitsandbytes_config = None,
    spqr_config = None,
    quanto_config = None,
    hqq_config = None,
    aqlm_config = None,
    vptq_config = None,
    ):
    
    quant_save_path = os.path.join(quant_save_path, model_id)
    
    if quantization_method == 'gptq':
        model = quantize_gptq(**gptq_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    elif quantization_method == 'awq':
        model = quantize_awq(**awq_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    elif quantization_method == 'bitsandbytes':
        model = quantize_bitsandbytes(**bitsandbytes_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    elif quantization_method == 'spqr':
        model = quantize_spqr(**spqr_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    elif quantization_method == 'quanto':
        model = quantize_quanto(**quanto_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    elif quantization_method == 'hqq':
        model = quantize_hqq(**hqq_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    elif quantization_method == 'aqlm':
        model = quantize_aqlm(**aqlm_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    elif quantization_method == 'vptq':
        model = quantize_vptq(**vptq_config)
        torch.save({
            'model': model.state_dict(),
            'model_id': model_id
            }, quant_save_path) 
    return

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization_method", type=str, required=True)
    parser.add_argument("--quant_save_path", type=str, required=True, default = "inf_prof/quantized_models/")
    parser.add_argument("--model_id", type=str, required=True, default = "athena")
    args = parser.parse_args()
    
    model_config = load_config("main/configs/model_config.json")
    model = Athena(**model_config)
    model, _ = load_weights_model(model, weights_path = "main/weights/athena.pt")
    quantized_model = quantize(model, quantization_method = args.quantization_method, quant_save_path = args.quant_save_path, model_id = args.model_id)