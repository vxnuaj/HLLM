import os
import sys
import yaml
import time
import torch
import json
import itertools
import logging
import torch.nn.functional as F

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers import StoppingCriteria, StoppingCriteriaList
from huggingface_hub import login as hf_login
from dotenv import load_dotenv

from quantize import quantize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
if os.getenv("HF_TOKEN"):
    hf_login(os.getenv("HF_TOKEN"))

@dataclass
class InferenceConfig:
    """Configuration for inference parameters with validation.

    Attributes:
        quantization_wbits (Optional[int]): Number of bits for weight quantization.
        quantization_method (Optional[str]): Method for weight quantization (e.g., 'gptq', 'awq').
        quantize_kvcache_method (Optional[str]): Method for KV cache quantization.
        quantize_kvcache (Optional[int]): Number of bits for KV cache quantization.
        model_execution_backend (str): Backend for model execution (e.g., 'eager', 'torch.compile').
        torch_compile_backend (Optional[str]): Backend for `torch.compile` (e.g., 'inductor').
        torch_compile_mode (Optional[str]): Mode for `torch.compile` (e.g., 'default').
        decoding (str): Decoding method (e.g., 'base', 'speculative').
    """
    quantization_wbits: Optional[int] = None
    quantization_method: Optional[str] = None
    quantize_kvcache_method: Optional[str] = None
    quantize_kvcache: Optional[int] = None
    model_execution_backend: str = "eager"
    torch_compile_backend: Optional[str] = None
    torch_compile_mode: Optional[str] = None
    decoding: str = "base"
    
    def __post_init__(self):
        """Performs post-initialization validation of configuration parameters.

        Raises:
            ValueError: If `quantization_wbits` is set without `quantization_method`,
                        or vice-versa. Similar validation for KV cache quantization.
        """
        if self.quantization_method and not self.quantization_wbits:
            raise ValueError("quantization_wbits must be set when quantization_method is specified")
        if self.quantization_wbits and not self.quantization_method:
            raise ValueError("quantization_method must be set when quantization_wbits is specified")
        if self.quantize_kvcache_method and not self.quantize_kvcache:
            raise ValueError("quantize_kvcache must be set when quantize_kvcache_method is specified")
        if self.quantize_kvcache and not self.quantize_kvcache_method:
            raise ValueError("quantize_kvcache_method must be set when quantize_kvcache is specified")
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the InferenceConfig object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the configuration.
        """
        return asdict(self)


class StoppingCriteriaSub(StoppingCriteria):
    """Custom stopping criteria for text generation.

    This class allows generation to stop when a specific sequence of tokens is encountered.

    Attributes:
        stops (List[List[int]]): A list of token ID sequences that, if found at the end of the generated text, will stop the generation.
    """
    
    def __init__(self, stops: List[List[int]] = None):
        """Initializes the StoppingCriteriaSub.

        Args:
            stops (List[List[int]], optional): A list of token ID sequences to stop generation. Defaults to None.
        """
        super().__init__()
        self.stops = stops or []
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        """Checks if generation should stop based on the presence of stop sequences.

        Args:
            input_ids (torch.LongTensor): The generated token IDs so far.
            scores (torch.FloatTensor): The scores of the generated tokens.

        Returns:
            bool: True if generation should stop, False otherwise.
        """
        for stop_seq in self.stops:
            if len(input_ids[0]) >= len(stop_seq):
                if input_ids[0][-len(stop_seq):].tolist() == stop_seq:
                    return True
        return False

def load_config(config_path: Union[str, Path], config_id: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration from a JSON or YAML file.

    Args:
        config_path (Union[str, Path]): Path to the configuration file.
        config_id (Optional[str], optional): Optional identifier for logging purposes. Defaults to None.

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration file format is unsupported.
        Exception: If loading the configuration fails for any other reason.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        if config_id:
            logger.info(f"Loaded {config_id} config from {config_path}")
        
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def load_tokenizer(tokenizer_path: str, tokenizer_type: str = "AutoTokenizer", **kwargs) -> Union[AutoTokenizer, PreTrainedTokenizerFast]:
    """Loads a tokenizer from HuggingFace or a local path.

    Args:
        tokenizer_path (str): Path to the tokenizer files or HuggingFace model ID.
        tokenizer_type (str, optional): The type of tokenizer to load ('AutoTokenizer' or 'PreTrainedTokenizerFast'). Defaults to "AutoTokenizer".
        **kwargs: Additional arguments to pass to the tokenizer's `from_pretrained` method.

    Returns:
        Union[AutoTokenizer, PreTrainedTokenizerFast]: The loaded tokenizer instance.

    Raises:
        Exception: If loading the tokenizer fails.
    """
    try:
        if tokenizer_type == "PreTrainedTokenizerFast":
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, **kwargs)
        else:  
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)
        
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        raise

def load_model(
    model_path: str,
    return_tokenizer: bool = False,
    tokenizer_config: Optional[Dict[str, Any]] = None,
    **model_kwargs
) -> Union[AutoModelForCausalLM, Tuple[AutoModelForCausalLM, AutoTokenizer]]:
    """Loads a pre-trained model and optionally its tokenizer from a specified path.

    Args:
        model_path (str): The path to the pre-trained model files or HuggingFace model ID.
        return_tokenizer (bool, optional): If True, the function will also return the tokenizer. Defaults to False.
        tokenizer_config (Optional[Dict[str, Any]], optional): Configuration for loading the tokenizer.
                                                                If None, `model_path` is used for tokenizer loading.
                                                                Defaults to None.
        **model_kwargs: Additional arguments to pass to the model's `from_pretrained` method.

    Returns:
        Union[AutoModelForCausalLM, Tuple[AutoModelForCausalLM, AutoTokenizer]]:
            The loaded model, or a tuple of (model, tokenizer) if `return_tokenizer` is True.

    Raises:
        Exception: If loading the model or tokenizer fails.
    """
    try:
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        if return_tokenizer:
            if tokenizer_config is None:
                tokenizer_config = {"tokenizer_path": model_path}
            
            tokenizer_path = tokenizer_config.pop("tokenizer_path", model_path)
            tokenizer = load_tokenizer(tokenizer_path, **tokenizer_config)
            return model, tokenizer
        
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

def create_combinations(sweep_config: Dict[str, List[Any]]) -> List[InferenceConfig]:
    """Generates all valid hyperparameter combinations from a given sweep configuration.

    This function takes a dictionary defining the sweep space, where each key maps
    to a list of possible values. It creates `InferenceConfig` objects for all
    permissible combinations, handling interdependencies (e.g., quantization bits
    and methods).

    Args:
        sweep_config (Dict[str, List[Any]]): A dictionary where keys are parameter names
                                            and values are lists of their possible settings.

    Returns:
        List[InferenceConfig]: A list of `InferenceConfig` objects, each representing a unique
                               and valid combination of inference parameters.
    """
    logger.info("Generating hyperparameter combinations from sweep configuration")
    
    quantization_wbits = [wbits for wbits in sweep_config.get('quantization_wbits', []) if wbits is not None]
    quantization_methods = [method for method in sweep_config.get('quantization_method', []) if method is not None]
    
    quantize_kvcache_methods = sweep_config.get('quantize_kvcache_method', [None])
    quantize_kvcache_bits = sweep_config.get('quantize_kvcache', [None])
    
    model_execution_backends = sweep_config.get('model_execution_backend', ['eager'])
    torch_compile_backends = sweep_config.get('torch_compile_backend', ['inductor'])
    torch_compile_modes = sweep_config.get('torch_compile_mode', ['default'])
    decoding_methods = sweep_config.get('decoding', ['base'])
    
    kv_combinations = len(quantize_kvcache_methods) * len(quantize_kvcache_bits)
    backend_combinations = len(model_execution_backends)
    decoding_combinations = len(decoding_methods)
    
    q_combinations = len(quantization_wbits) * len(quantization_methods) * kv_combinations * backend_combinations * decoding_combinations
    neither_combinations = kv_combinations * backend_combinations * decoding_combinations
    base_combinations = q_combinations + neither_combinations
    
    torch_compile_expansion = len(torch_compile_backends) * len(torch_compile_modes)
    total_combinations = base_combinations * torch_compile_expansion
    logger.info(f"Quantization combinations: {q_combinations}")
    logger.info(f"Neither combinations: {neither_combinations}")
    logger.info(f"Base combinations: {base_combinations}")
    logger.info(f"Torch.compile expansion factor: {torch_compile_expansion}")
    logger.info(f"Total combinations: {total_combinations}")
    configs = []
    
    with tqdm(total=total_combinations, desc="Generating configurations") as pbar:
        for q_wbits, q_method in itertools.product(quantization_wbits, quantization_methods):
            for kv_method, kv_bits in itertools.product(quantize_kvcache_methods, quantize_kvcache_bits):
                if (kv_method is None) != (kv_bits is None):
                    continue
                for backend in model_execution_backends:
                    for decoding in decoding_methods:
                        for compile_backend in torch_compile_backends:
                            for compile_mode in torch_compile_modes:
                                try:
                                    config = InferenceConfig(
                                        quantization_wbits=q_wbits,
                                        quantization_method=q_method,
                                        quantize_kvcache_method=kv_method,
                                        quantize_kvcache=kv_bits,
                                        model_execution_backend=backend,
                                        torch_compile_backend=compile_backend,
                                        torch_compile_mode=compile_mode,
                                        decoding=decoding
                                    )
                                    configs.append(config)
                                    pbar.update(1)
                                except ValueError as e:
                                    logger.warning(f"Skipping invalid config: {e}")
                                    pbar.update(1)
        
        for kv_method, kv_bits in itertools.product(quantize_kvcache_methods, quantize_kvcache_bits):
            if (kv_method is None) != (kv_bits is None):
                continue
            for backend in model_execution_backends:
                for decoding in decoding_methods:
                    for compile_backend in torch_compile_backends:
                        for compile_mode in torch_compile_modes:
                            try:
                                config = InferenceConfig(
                                    quantization_wbits=None,
                                    quantization_method=None,
                                    quantize_kvcache_method=kv_method,
                                    quantize_kvcache=kv_bits,
                                    model_execution_backend=backend,
                                    torch_compile_backend=compile_backend,
                                    torch_compile_mode=compile_mode,
                                    decoding=decoding
                                )
                                configs.append(config)
                                pbar.update(1)
                            except ValueError as e:
                                logger.warning(f"Skipping invalid config: {e}")
                                pbar.update(1)
    
    logger.info(f"Generated {len(configs)} valid configurations")
    return configs

def apply_quantization(
    model: torch.nn.Module,
    method: str,
    wbits: int,
    calibration_data: Optional[torch.Tensor] = None
) -> Tuple[torch.nn.Module, Optional[Any]]:
    """Applies a specified quantization method to the model.

    Args:
        model (torch.nn.Module): The PyTorch model to quantize.
        method (str): The quantization method to apply (e.g., 'gptq', 'awq').
        wbits (int): The number of bits for weight quantization.
        calibration_data (Optional[torch.Tensor], optional): Optional data for calibration during quantization. Defaults to None.

    Returns:
        Tuple[torch.nn.Module, Optional[Any]]: A tuple containing the quantized model and any associated quantizers.

    Raises:
        Exception: If the quantization process fails.
    """
    logger.info(f"Applying quantization: {method} with {wbits} bits")
    
    try:
        return quantize(
            model=model,
            method=method,
            wbits=wbits,
            calibration_data=calibration_data
        )
    except Exception as e:
        logger.error(f"Quantization failed with {method}: {e}")
        raise


def compile_model(
    model: torch.nn.Module,
    backend: str = "inductor",
    mode: str = "default"
) -> torch.nn.Module:
    """Compiles a PyTorch model using `torch.compile`.

    Args:
        model (torch.nn.Module): The PyTorch model to compile.
        backend (str, optional): The backend for `torch.compile` (e.g., 'inductor', 'eager'). Defaults to "inductor".
        mode (str, optional): The compilation mode (e.g., 'default', 'reduce-overhead'). Defaults to "default".

    Returns:
        torch.nn.Module: The compiled model.

    Raises:
        Exception: If model compilation fails.
    """
    logger.info(f"Compiling model with backend={backend}, mode={mode}")
    
    try:
        return torch.compile(model, backend=backend, mode=mode)
    except Exception as e:
        logger.error(f"Model compilation failed: {e}")
        raise


def apply_config_to_model(
    model: torch.nn.Module,
    config: InferenceConfig,
    calibration_data: Optional[torch.Tensor] = None
) -> Tuple[torch.nn.Module, Optional[Any]]:
    """Applies a given inference configuration to a model, including quantization and compilation.

    Args:
        model (torch.nn.Module): The base model to configure.
        config (InferenceConfig): The inference configuration to apply.
        calibration_data (Optional[torch.Tensor], optional): Optional calibration data for quantization. Defaults to None.

    Returns:
        Tuple[torch.nn.Module, Optional[Any]]: A tuple containing the configured model and any associated quantizers.
    """
    logger.info(f"Applying configuration: {config}")
    
    quantizers = None
    
    if config.quantization_method and config.quantization_wbits:
        model, quantizers = apply_quantization(
            model,
            config.quantization_method,
            config.quantization_wbits,
            calibration_data
        )
    
    if config.model_execution_backend == "torch.compile":
        model = compile_model(
            model,
            config.torch_compile_backend or "inductor",
            config.torch_compile_mode or "default"
        )
    
    return model, quantizers


def save_results(results: List[Dict], output_path: Union[str, Path]) -> None:
    """Saves a list of sweep results to a JSON file.

    Args:
        results (List[Dict]): A list of dictionaries, where each dictionary represents the results of an inference configuration.
        output_path (Union[str, Path]): The path to the file where the results will be saved.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")


def run_inference_sweep(
    model_path: str,
    sweep_config_path: str,
    output_path: str,
    tokenizer_path: Optional[str] = None,
    calibration_data_path: Optional[str] = None,
    assistant_model_path: Optional[str] = None,
    X_tensor_path: Optional[str] = None,
    y_tensor_path: Optional[str] = None,
    inference_batch_size: int = 16,
    inference_seed_seq_len: int = 10,
    top_k: int = 50,
    top_p: float = 0.9,
    inference_iter: int = 10,
    max_new_tokens: int = 512,
    **inference_kwargs
) -> List[Dict]:
    """Runs a comprehensive inference sweep over various model configurations.

    This function loads a base model, generates all combinations of inference
    configurations from a sweep configuration file, applies these configurations
    (including quantization and compilation), and measures inference times and losses.
    Results are saved to a specified output path.

    Args:
        model_path (str): Path to the base model to be used for the sweep.
        sweep_config_path (str): Path to the YAML or JSON file defining the sweep configuration.
        output_path (str): Path to the file where the sweep results will be saved.
        tokenizer_path (Optional[str], optional): Path to the tokenizer. If None, `model_path` is used. Defaults to None.
        calibration_data_path (Optional[str], optional): Path to calibration data for quantization. Defaults to None.
        assistant_model_path (Optional[str], optional): Path to an assistant model for speculative decoding. Defaults to None.
        X_tensor_path (Optional[str], optional): Path to the input tensor (X) for inference. Defaults to None.
        y_tensor_path (Optional[str], optional): Path to the target tensor (y) for loss calculation. Defaults to None.
        inference_batch_size (int, optional): Batch size to use during inference. Defaults to 16.
        inference_seed_seq_len (int, optional): Length of the seed sequence for generation. Defaults to 10.
        top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k sampling. Defaults to 50.
        top_p (float, optional): The cumulative probability for top-p (nucleus) sampling. Defaults to 0.9.
        inference_iter (int, optional): Number of inference iterations to average for timing. Defaults to 10.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 512.
        **inference_kwargs: Additional keyword arguments to pass to the model's `generate` method.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary contains the configuration
                    and measured metrics for a single inference run.
    """
    
    sweep_config = load_config(sweep_config_path, "sweep")
    configs = create_combinations(sweep_config)
    
    base_model = load_model(model_path)
#    tokenizer = load_tokenizer(tokenizer_path or model_path)
    
    assistant_model = None
    if assistant_model_path:
        assistant_model = load_model(assistant_model_path)
    
    X = None
    y = None
    if X_tensor_path:
        X = torch.load(X_tensor_path, map_location='cpu')
    if y_tensor_path:
        y = torch.load(y_tensor_path, map_location='cpu')
    
    calibration_data = None
    if calibration_data_path:
        calibration_data = torch.load(calibration_data_path, map_location='cpu')
    
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[[2]])])
    
    results = []
    
    for i, config in enumerate(tqdm(configs, desc="Running inference sweep")):
        try:
            configured_model, quantizers = apply_config_to_model(
                base_model, config, calibration_data
            )
            
            if config.model_execution_backend == "torch.compile":
                logger.info(f"Warming up torch.compile for config {i}")
                for _ in range(10):  # Warmup iterations
                    if X is not None:
                        configured_model.generate(
                            X, 
                            max_new_tokens=max_new_tokens, 
                            stopping_criteria=stopping_criteria, 
                            output_scores=True, 
                            top_k=top_k, 
                            top_p=top_p
                        )
            inf_time_list = []
            loss_list = []
            
            if config.decoding == "base":
                for _ in range(inference_iter):
                    start_time = time.perf_counter()
                    
                    outputs = configured_model.generate(
                            X, 
                            max_new_tokens=max_new_tokens, 
                            stopping_criteria=stopping_criteria,
                            output_scores=True, 
                            top_k=top_k,
                            top_p=top_p
                        )
                    
                    end_time = time.perf_counter()
                    inf_time_list.append(end_time - start_time)
                    
                    loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), y.view(-1)).item()
                    loss_list.append(loss)
            
            elif config.decoding == "speculative":
                for _ in range(inference_iter):
                    start_time = time.perf_counter()
                    
                    outputs = configured_model.generate(
                        X, 
                        max_new_tokens=max_new_tokens, 
                        stopping_criteria=stopping_criteria,
                        output_scores=True, 
                        top_p=top_p,
                        top_k=top_k,
                        assistant_model=assistant_model
                    )
                    
                    end_time = time.perf_counter()
                    inf_time_list.append(end_time - start_time)
                    loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), y.view(-1)).item()
                    loss_list.append(loss)
                    
            avg_inference_time = sum(inf_time_list) / len(inf_time_list) if inf_time_list else 0.0
            avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
            
            result = {
                'config_id': i,
                'config': config.to_dict(),
                'avg_inference_time': avg_inference_time,
                'avg_loss': avg_loss,
                'inference_times': inf_time_list,
                'losses': loss_list,
                'success': True,
                'error': None
            }
            
            logger.info(f"Config {i}: avg_time={avg_inference_time:.4f}s, avg_loss={avg_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Configuration {i} failed: {e}")
            result = {
                'config_id': i,
                'config': config.to_dict(),
                'avg_inference_time': None,
                'avg_loss': None,
                'inference_times': [],
                'losses': [],
                'success': False,
                'error': str(e)
            }
        
        results.append(result)
    
    save_results(results, output_path)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference sweep")
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--sweep_config", required=True, help="Path to sweep config")
    parser.add_argument("--output_path", required=True, help="Path to save results")
    parser.add_argument("--tokenizer_path", help="Path to tokenizer")
    parser.add_argument("--calibration_data", help="Path to calibration data")
    parser.add_argument("--assistant_model_path", help="Path to assistant model for speculative decoding")
    parser.add_argument("--X_tensor_path", help="Path to input tensor for inference")
    parser.add_argument("--y_tensor_path", help="Path to target tensor for loss calculation")
    parser.add_argument("--inference_batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--inference_seed_seq_len", type=int, default=10, help="Seed sequence length")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--inference_iter", type=int, default=10, help="Number of inference iterations for averaging")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    run_inference_sweep(
        model_path=args.model_path,
        sweep_config_path=args.sweep_config,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        calibration_data_path=args.calibration_data,
        assistant_model_path=args.assistant_model_path,
        X_tensor_path=args.X_tensor_path,
        y_tensor_path=args.y_tensor_path,
        inference_batch_size=args.inference_batch_size,
        inference_seed_seq_len=args.inference_seed_seq_len,
        top_k=args.top_k,
        top_p=args.top_p,
        inference_iter=args.inference_iter,
        max_new_tokens=args.max_new_tokens
        )