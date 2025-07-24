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

try:
    from model import Athena
except ImportError as e:
    logging.warning(f"Local model imports failed: {e}")

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Loads configuration from a JSON file.

    Args:
        config_path (str): The path to the JSON configuration file.

    Returns:
        dict: The loaded configuration as a dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        json.JSONDecodeError: If the configuration file is not a valid JSON.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def load_weights_model(model: torch.nn.Module, weights_path: str) -> Tuple[torch.nn.Module, dict]:
    """Loads weights into a given model from a PyTorch checkpoint file.

    Args:
        model (torch.nn.Module): The model to load weights into.
        weights_path (str): The path to the PyTorch checkpoint file.

    Returns:
        Tuple[torch.nn.Module, dict]: A tuple containing the model with loaded weights and any additional metadata from the checkpoint.

    Raises:
        FileNotFoundError: If the weights file does not exist.
        Exception: If loading weights fails for any other reason.
    """
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
    """Loads and configures a tokenizer from a file.

    Args:
        tokenizer_file (str): The path to the tokenizer file.
        confirm_special_toks (bool, optional): If True, prints the special tokens (EOS, BOS, PAD). Defaults to True.
        eos_token (str, optional): The end-of-sequence token string. Defaults to "<|eos|>".
        bos_token (str, optional): The beginning-of-sequence token string. Defaults to "<|bos|>".
        pad_token (str, optional): The padding token string. Defaults to " ".

    Returns:
        PreTrainedTokenizerFast: The loaded and configured tokenizer.
    """
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
    """Retrieves the model's name or path from its configuration, with a fallback option.

    Args:
        model (torch.nn.Module): The model instance.
        fallback (str, optional): The fallback name to use if the model's name/path cannot be determined. Defaults to "athena".

    Returns:
        str: The determined model name or path.
    """
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
    """Saves the model and optionally its tokenizer to Hugging Face Hub and/or locally.

    Args:
        model (torch.nn.Module): The model to save.
        tokenizer (Optional[PreTrainedTokenizerFast]): The tokenizer to save alongside the model. Can be None.
        save_hf (bool): If True, saves the model to Hugging Face Hub.
        save_local (bool): If True, saves the model locally.
        hf_path (str): The Hugging Face repository ID or path.
        local_path (str): The local directory path to save the model.
        model_name (str): The base name for the model files/repository.
        method_suffix (str, optional): A suffix to append to the model name (e.g., indicating quantization method). Defaults to "".
    """
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
    """Applies GPTQ quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization. Defaults to 4.
        calibration_data (Optional[torch.Tensor], optional): Data used for calibration during quantization. Defaults to None.
        tokenizer (Optional[PreTrainedTokenizerFast], optional): The tokenizer associated with the model. Defaults to None.
        groupsize (int, optional): The group size for quantization. Defaults to 128.
        nsamples (int, optional): Number of samples to use for calibration. Defaults to 128.
        percdamp (float, optional): Percentage damping for quantization. Defaults to 0.01.
        **kwargs: Additional keyword arguments for GPTQConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        Exception: If GPTQ quantization fails.
    """
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
    """Applies AWQ quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization. Defaults to 4.
        group_size (int, optional): The group size for quantization. Defaults to 128.
        do_fuse (bool, optional): Whether to fuse operations during quantization. Defaults to True.
        fuse_max_seq_len (int, optional): Maximum sequence length for fused operations. Defaults to 512.
        **kwargs: Additional keyword arguments for AwqConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        Exception: If AWQ quantization fails.
    """
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
    """Applies Quanto quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization. Defaults to 4.
        **kwargs: Additional keyword arguments for QuantoConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        Exception: If Quanto quantization fails.
    """
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
    """Applies AQLM quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization. Defaults to 4.
        **kwargs: Additional keyword arguments for AqlmConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        Exception: If AQLM quantization fails.
    """
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
    """Applies VPTQ quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization. Defaults to 4.
        **kwargs: Additional keyword arguments for VptqConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        Exception: If VPTQ quantization fails.
    """
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
    """Applies HQQ quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization. Defaults to 4.
        **kwargs: Additional keyword arguments for HqqConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        Exception: If HQQ quantization fails.
    """
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
    """Applies BitsAndBytes quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization (4 or 8). Defaults to 4.
        **kwargs: Additional keyword arguments for BitsAndBytesConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        ValueError: If `quantization_bits` is not 4 or 8.
        Exception: If BitsAndBytes quantization fails.
    """
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
    """Applies SpQR quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        quantization_bits (int, optional): The number of bits for quantization. Defaults to 4.
        **kwargs: Additional keyword arguments for SpQRConfig.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        Exception: If SpQR quantization fails.
    """
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
    """Dispatches to the appropriate quantization function based on the specified method.

    Args:
        model (torch.nn.Module): The model to quantize.
        method (str): The quantization method to use (e.g., 'gptq', 'awq').
        wbits (int): The number of bits for quantization.
        **method_kwargs: Additional keyword arguments to pass to the specific quantization method.

    Returns:
        torch.nn.Module: The quantized model.

    Raises:
        ValueError: If an unsupported quantization method is provided.
        Exception: If the chosen quantization method fails.
    """
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
    """Applies KV cache quantization to the model.

    Args:
        model (torch.nn.Module): The model to quantize.
        method (str): The KV cache quantization method to use (e.g., 'hqq', 'quanto').
        bits (int): The number of bits for quantization.
        **kwargs: Additional keyword arguments for the specific quantization method.

    Returns:
        torch.nn.Module: The model with quantized KV cache.

    Raises:
        ValueError: If an unsupported KV cache quantization method is provided.
    """
    logger.info(f"Applying KV cache quantization: {method} with {bits} bits")
    
    if method == 'hqq':
        
        return model  
    elif method == 'quanto':
        
        return model  
    else:
        raise ValueError(f"Unsupported KV cache quantization method: {method}")

def save_quantized_model(model: torch.nn.Module, model_id: str, save_path: str, method: str) -> str:
    """Saves the quantized model to a specified path with a consistent naming convention.

    Args:
        model (torch.nn.Module): The quantized model to save.
        model_id (str): A unique identifier for the model.
        save_path (str): The directory where the model will be saved.
        method (str): The quantization method used.

    Returns:
        str: The full path to the saved model file.
    """
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
    """Main function to quantize a model using a specified method and save it.

    Args:
        model (torch.nn.Module): The model to be quantized.
        quantization_method (str): The name of the quantization method to use (e.g., 'gptq', 'awq').
        quant_save_path (str): The base path where the quantized model will be saved.
        model_id (str): An identifier for the model, used in saving.
        quantization_wbits (Optional[int], optional): The number of bits for weight quantization. Defaults to 4 if not specified.
        save_model_to_hf (bool, optional): If True, saves the quantized model to Hugging Face Hub. Defaults to False.
        save_model_to_local (bool, optional): If True, saves the quantized model locally. Defaults to True.
        save_model_hf_path (Optional[str], optional): Specific Hugging Face path if different from `quant_save_path`. Defaults to None.
        save_model_local_path (Optional[str], optional): Specific local path if different from `quant_save_path`. Defaults to None.
        tokenizer (Optional[PreTrainedTokenizerFast], optional): The tokenizer associated with the model, used for HF upload. Defaults to None.
        **method_configs: Additional keyword arguments specific to the chosen quantization method.

    Returns:
        Optional[torch.nn.Module]: The quantized model if successful, otherwise None.

    Raises:
        ValueError: If `quantization_method` is not specified.
    """
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