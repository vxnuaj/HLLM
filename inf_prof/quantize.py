import os
import sys
import json
import logging
import torch
import argparse
from typing import Union, Optional, Tuple, Any, List

try:
    from transformers import (
        PreTrainedTokenizerFast, AutoModelForCausalLM, Tokenizer, 
        AwqConfig, QuantoConfig, VptqConfig, BitsAndBytesConfig, 
        HqqConfig, SpQRConfig, GPTQConfig, AqlmConfig
    )
except ImportError as e:
    logging.warning(f"Some transformers imports failed: {e}")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'package', 'gptq'))

try:
    from gptq import GPTQ
    from model import Athena
except ImportError as e:
    logging.warning(f"Local model imports failed: {e}")

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def load_weights_model(model: torch.nn.Module, weights_path: str) -> Tuple[torch.nn.Module, dict]:
    """Load weights into model from checkpoint"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            metadata = {k: v for k, v in checkpoint.items() if k != 'model'}
        else:
            model.load_state_dict(checkpoint)
            metadata = {}
        
        logger.info(f"Successfully loaded weights from {weights_path}")
        return model, metadata
    except Exception as e:
        logger.error(f"Failed to load weights from {weights_path}: {e}")
        raise

def load_tokenizer(
    tokenizer_file: str, 
    confirm_special_toks: bool = True, 
    eos_token: str = "<|eos|>", 
    bos_token: str = "<|bos|>", 
    pad_token: str = " "
) -> PreTrainedTokenizerFast:
    """Load and configure tokenizer"""
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_file,
        eos_token=eos_token,
        bos_token=bos_token,
        pad_token=pad_token
    )
    
    if confirm_special_toks:
        print(f"EOS: {tokenizer.eos_token}")
        print(f"BOS: {tokenizer.bos_token}")
        print(f"PAD: {tokenizer.pad_token}")
    
    return tokenizer

def get_model_name_or_path(model: torch.nn.Module, fallback: str = "athena") -> str:
    """Get model name or path with fallback"""
    if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
        return model.config.name_or_path
    elif hasattr(model, 'name_or_path'):
        return model.name_or_path
    elif hasattr(model, '_name_or_path'):
        return model._name_or_path
    else:
        logger.warning(f"Could not determine model name, using fallback: {fallback}")
        return fallback

def save_model_hub_and_local(
    model: torch.nn.Module,
    tokenizer: Optional[PreTrainedTokenizerFast],
    save_hf: bool,
    save_local: bool,
    hf_path: str,
    local_path: str,
    model_name: str,
    method_suffix: str = ""
):
    """Unified model saving function"""
    if not (save_hf or save_local):
        return
    
    model_name_with_suffix = f"{model_name}-{method_suffix}" if method_suffix else model_name
    
    if save_hf:
        repo_id = f"{hf_path}/{model_name_with_suffix}" if "/" not in hf_path else hf_path
        logger.info(f"Saving model to HuggingFace Hub: {repo_id}")
        
        model.push_to_hub(
            repo_id=repo_id,
            commit_message=f"Upload quantized {model_name} model",
            private=False
        )
        
        if tokenizer:
            tokenizer.push_to_hub(
                repo_id=repo_id,
                commit_message=f"Upload tokenizer for {model_name}",
                private=False
            )
    
    if save_local:
        local_file_path = os.path.join(local_path, model_name_with_suffix)
        logger.info(f"Saving model locally: {local_file_path}")
        torch.save(model.state_dict(), local_file_path)


def quantize_gptq(
    model: torch.nn.Module,
    quantization_bits: int = 4,
    calibration_data: Optional[torch.Tensor] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    groupsize: int = 128,
    nsamples: int = 128,
    percdamp: float = 0.01,
    **kwargs
) -> torch.nn.Module:
    """Apply GPTQ quantization"""
    logger.info('Initializing GPTQ Quantization.')
    
    try:
        
        if isinstance(calibration_data, (list, torch.tensor, torch.Tensor)):
            if tokenizer:
                calibration_data = tokenizer.batch_decode(calibration_data)
            if isinstance(calibration_data, list):
                calibration_data = torch.tensor(calibration_data)
        
        gptq_config = GPTQConfig(
            bits=quantization_bits,
            dataset=calibration_data,
            tokenizer=tokenizer,
            group_size=groupsize,
            batch_size=nsamples,
            damp_percent=percdamp,
        )
        
        model_name = get_model_name_or_path(model)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=gptq_config,
        )
        
        return quantized_model
        
    except Exception as e:
        logger.error(f"GPTQ quantization failed: {e}")
        raise

def quantize_awq(
    model: torch.nn.Module,
    quantization_bits: int = 4,
    group_size: int = 128,
    do_fuse: bool = True,
    fuse_max_seq_len: int = 512,
    **kwargs
) -> torch.nn.Module:
    """Apply AWQ quantization"""
    logger.info("Initializing AWQ Quantization")
    
    try:
        from awq import AWQLinerVersion
        
        awq_config = AwqConfig(
            bits=quantization_bits,
            group_size=group_size,
            version=AWQLinerVersion.GEMV,
            do_fuse=do_fuse,
            fuse_max_seq_len=fuse_max_seq_len,
        )
        
        model_name = get_model_name_or_path(model)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=awq_config,
            device_map='auto'
        )
        
        return quantized_model
        
    except Exception as e:
        logger.error(f"AWQ quantization failed: {e}")
        raise

def quantize_quanto(model: torch.nn.Module, quantization_bits: int = 4, **kwargs) -> torch.nn.Module:
    """Apply Quanto quantization"""
    logger.info("Initializing Quanto Quantization")
    
    try:
        quanto_config = QuantoConfig(weights=f"int{quantization_bits}", **kwargs)
        model_name = get_model_name_or_path(model)
        
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quanto_config,
            device_map='auto'
        )
        return quantized_model
        
    except Exception as e:
        logger.error(f"Quanto quantization failed: {e}")
        raise

def quantize_aqlm(model: torch.nn.Module, quantization_bits: int = 4, **kwargs) -> torch.nn.Module:
    """Apply AQLM quantization"""
    logger.info("Initializing AQLM Quantization")
    
    try:
        aqlm_config = AqlmConfig(nbits_act=quantization_bits, **kwargs)
        model_name = get_model_name_or_path(model)
        
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=aqlm_config,
            device_map='auto'
        )
        return quantized_model
        
    except Exception as e:
        logger.error(f"AQLM quantization failed: {e}")
        raise

def quantize_vptq(model: torch.nn.Module, quantization_bits: int = 4, **kwargs) -> torch.nn.Module:
    """Apply VPTQ quantization"""
    logger.info("Initializing VPTQ Quantization")
    
    try:
        vptq_config = VptqConfig(bits=quantization_bits, **kwargs)
        model_name = get_model_name_or_path(model)
        
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=vptq_config,
            device_map='auto'  
        )
        return quantized_model
        
    except Exception as e:
        logger.error(f"VPTQ quantization failed: {e}")
        raise

def quantize_hqq(model: torch.nn.Module, quantization_bits: int = 4, **kwargs) -> torch.nn.Module:
    """Apply HQQ quantization"""
    logger.info("Initializing HQQ Quantization")
    
    try:
        hqq_config = HqqConfig(nbits=quantization_bits, **kwargs)
        model_name = get_model_name_or_path(model)
        
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=hqq_config,
            device_map='auto'
        )
        return quantized_model
        
    except Exception as e:
        logger.error(f"HQQ quantization failed: {e}")
        raise

def quantize_bitsandbytes(model: torch.nn.Module, quantization_bits: int = 4, **kwargs) -> torch.nn.Module:
    """Apply BitsAndBytes quantization"""
    logger.info("Initializing BitsAndBytes Quantization")
    
    try:
        if quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                **kwargs
            )
        elif quantization_bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, **kwargs)
        else:
            raise ValueError(f"BitsAndBytes only supports 4 or 8 bits, got {quantization_bits}")
        
        model_name = get_model_name_or_path(model)
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto'
        )
        return quantized_model
        
    except Exception as e:
        logger.error(f"BitsAndBytes quantization failed: {e}")
        raise

def quantize_spqr(model: torch.nn.Module, quantization_bits: int = 4, **kwargs) -> torch.nn.Module:
    """Apply SpQR quantization"""
    logger.info("Initializing SpQR Quantization")
    
    try:
        spqr_config = SpQRConfig(bits=quantization_bits, **kwargs)
        model_name = get_model_name_or_path(model)
        
        quantized_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=spqr_config,
            device_map='auto'
        )
        return quantized_model
        
    except Exception as e:
        logger.error(f"SpQR quantization failed: {e}")
        raise


QUANTIZATION_METHODS = {
    'gptq': quantize_gptq,
    'awq': quantize_awq,
    'quanto': quantize_quanto,
    'aqlm': quantize_aqlm,
    'vptq': quantize_vptq,
    'hqq': quantize_hqq,
    'bitsandbytes': quantize_bitsandbytes,
    'spqr': quantize_spqr
}

def quantize_model_wbits(
    model: torch.nn.Module,
    method: str,
    wbits: int,
    **method_kwargs
) -> torch.nn.Module:
    """Main quantization dispatcher"""
    if method not in QUANTIZATION_METHODS:
        available = ', '.join(QUANTIZATION_METHODS.keys())
        raise ValueError(f"Unsupported method '{method}'. Available: {available}")
    
    logger.info(f"Applying {method} quantization with {wbits} bits")
    
    try:
        quantize_fn = QUANTIZATION_METHODS[method]
        return quantize_fn(model, quantization_bits=wbits, **method_kwargs)
    except Exception as e:
        logger.error(f"Quantization with {method} failed: {e}")
        raise

def quantize_kvcache(model: torch.nn.Module, method: str, bits: int, **kwargs) -> torch.nn.Module:
    """Apply KV cache quantization"""
    logger.info(f"Applying KV cache quantization: {method} with {bits} bits")
    
    if method == 'hqq':
        
        return model  
    elif method == 'quanto':
        
        return model  
    else:
        raise ValueError(f"Unsupported KV cache quantization method: {method}")

def save_quantized_model(model: torch.nn.Module, model_id: str, save_path: str, method: str) -> str:
    """Save quantized model with consistent naming"""
    full_path = os.path.join(save_path, f"{model_id}_{method}.pt")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    torch.save({
        'model': model.state_dict(),
        'model_id': model_id,
        'quantization_method': method
    }, full_path)
    
    logger.info(f"Saved {method} quantized model to {full_path}")
    return full_path

def quantize(
    model: torch.nn.Module,
    quantization_method: str,
    quant_save_path: str,
    model_id: str,
    quantization_wbits: Optional[int] = None,
    save_model_to_hf: bool = False,
    save_model_to_local: bool = True,
    save_model_hf_path: Optional[str] = None,
    save_model_local_path: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
    **method_configs
) -> Optional[torch.nn.Module]:
    """Main quantization function with unified interface"""
    if not quantization_method:
        raise ValueError("quantization_method must be specified")
    
    if not quantization_wbits:
        logger.warning("No quantization_wbits specified, using default of 4")
        quantization_wbits = 4
    
    try:
        
        method_key = f"{quantization_method}_config"
        method_config = method_configs.get(method_key, {})
        
        
        quantized_model = quantize_model_wbits(
            model=model,
            method=quantization_method,
            wbits=quantization_wbits,
            **method_config
        )
        
        
        if save_model_to_local or save_model_to_hf:
            save_model_hub_and_local(
                model=quantized_model,
                tokenizer=tokenizer,
                save_hf=save_model_to_hf,
                save_local=save_model_to_local,
                hf_path=save_model_hf_path or quant_save_path,
                local_path=save_model_local_path or quant_save_path,
                model_name=model_id,
                method_suffix=quantization_method
            )
        
        logger.info(f"Successfully quantized model using {quantization_method}")
        return quantized_model
        
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Athena model")
    parser.add_argument("--quantization_method", type=str, required=True,
                       choices=list(QUANTIZATION_METHODS.keys()),
                       help="Quantization method to use")
    parser.add_argument("--quant_save_path", type=str, required=True, 
                       default="inf_prof/quantized_models/",
                       help="Path to save quantized model")
    parser.add_argument("--model_id", type=str, required=True, default="athena",
                       help="Model identifier")
    parser.add_argument("--quantization_wbits", type=int, default=4,
                       help="Quantization bit width")
    parser.add_argument("--save_model_to_hf", action='store_true',
                       help="Save model to HuggingFace Hub")
    parser.add_argument("--save_model_to_local", action='store_true', default=True,
                       help="Save model locally")
    parser.add_argument("--save_model_hf_path", type=str,
                       help="HuggingFace Hub path")
    parser.add_argument("--save_model_local_path", type=str,
                       help="Local save path")
    
    args = parser.parse_args()
    
    try:
        
        model_config = load_config("main/configs/model_config.json")
        model = Athena(**model_config)
        model, _ = load_weights_model(model, weights_path="main/weights/athena.pt")
        
        
        quantized_model = quantize(
            model=model,
            quantization_method=args.quantization_method,
            quant_save_path=args.quant_save_path,
            model_id=args.model_id,
            quantization_wbits=args.quantization_wbits,
            save_model_to_hf=args.save_model_to_hf,
            save_model_to_local=args.save_model_to_local,
            save_model_hf_path=args.save_model_hf_path,
            save_model_local_path=args.save_model_local_path
        )
        
        if quantized_model is not None:
            logger.info("Quantization completed successfully")
        else:
            logger.error("Quantization failed")
            
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise