## `inf_prof/`: Inference and Quantization Profiling

This directory contains scripts and configurations for profiling inference performance and applying various quantization techniques to Large Language Models.

### Quantization (`quantize.py`)

The `quantize.py` script allows you to quantize a pre-trained model using different methods (e.g., GPTQ, AWQ, BitsAndBytes). It supports specifying the quantization bit-width and saving the quantized model locally or uploading it to the Hugging Face Hub.

**Usage:**

To quantize a model, run the `quantize.py` script with the desired arguments. You can specify the quantization method, bit-width, model ID, and save paths.

```bash
# Example: Quantize a model using GPTQ with 4 bits
python inf_prof/quantize.py \
    --quantization_method gptq \
    --quantization_wbits 4 \
    --model_id your_model_name \
    --quant_save_path inf_prof/quantized_models/

# Example: Quantize using BitsAndBytes with 8 bits and save to Hugging Face
python inf_prof/quantize.py \
    --quantization_method bitsandbytes \
    --quantization_wbits 8 \
    --model_id your_model_name \
    --save_model_to_hf \
    --save_model_hf_path your_hf_org/your_repo
```

**Available Quantization Methods:**

All quantization methods are now integrated and handled via the Hugging Face `transformers` library. Refer to the `QUANTIZATION_METHODS` dictionary in `inf_prof/quantize.py` for a list of supported methods (e.g., `gptq`, `awq`, `quanto`, `aqlm`, `vptq`, `hqq`, `bitsandbytes`, `spqr`).

### Inference Sweep (`sweep_utils.py`)

The `sweep_utils.py` script (which is typically invoked by `inf_prof/sweep.py`) is used to run comprehensive inference sweeps over various model configurations. It allows you to test different quantization methods, model execution backends (e.g., `torch.compile`), and decoding strategies.

**Configuration:**

The sweep parameters are defined in `inf_prof/search_space.yaml`. Modify this file to define the combinations of hyperparameters you want to test.

**Usage:**

To run an inference sweep, you would typically execute `inf_prof/sweep.py` (which internally calls `run_inference_sweep` from `sweep_utils.py`).

```bash
# Example: Run an inference sweep
python inf_prof/sweep.py
```

This script will load the `search_space.yaml`, generate all valid combinations, apply them to the model, and measure inference times and losses. Results are saved to `inf_prof/results/`.

**Key Parameters in `inf_prof/sweep.py` (passed to `run_inference_sweep`):**

-   `model_path`: Path to the base model.
-   `sweep_config_path`: Path to `search_space.yaml`.
-   `output_path`: Directory to save sweep results.
-   `tokenizer_path`: Path to the tokenizer (optional).
-   `calibration_data_path`: Path to calibration data for quantization (optional).
-   `assistant_model_path`: Path to an assistant model for speculative decoding (optional).
-   `X_tensor_path`, `y_tensor_path`: Paths to input/target tensors for evaluation (optional).
-   `inference_batch_size`, `inference_iter`, `max_new_tokens`, `top_k`, `top_p`: Inference generation parameters.

