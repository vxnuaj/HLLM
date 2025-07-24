# HLLM: High-Performance Large Language Model Training and Optimization Framework

HLLM is a comprehensive framework designed for efficient training, profiling, and quantization of Large Language Models (LLMs). It supports distributed training, various quantization techniques, and performance profiling.

## Features

-   **Distributed LLM Training**: Scalable training using PyTorch's distributed capabilities.
-   **Flexible Configuration**: Easily configurable training, model, optimizer, and scheduler parameters via JSON files.
-   **Inference Optimization**: Support for various quantization methods (GPTQ, AWQ, etc.) and model execution backends (torch.compile, ONNX Runtime, TensorRT, vLLM).
-   **Performance Profiling**: Tools for analyzing training and inference performance, including hyperparameter sweeps.
-   **Hugging Face Integration**: Utilities for converting models and uploading to Hugging Face Hub.

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/vxnuaj/HLLM.git
    cd HLLM
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For AWQ quantization, refer to `inf_prof/README.md` for additional setup steps.*

## Usage

### Training

To train a model, first ensure your data is preprocessed (if required). The preprocessing script `preprocess.py` prepares your dataset. After preprocessing, you can initiate training using `train.py` with `torchrun` for distributed execution.

```bash
# From the project root directory
python main/preprocess.py
torchrun --standalone --nproc-per-node=<num_gpus> --nnodes=1 main/train.py
```
Replace `<num_gpus>` with the number of GPUs you want to use for training.

### Configuration

All core configurations for training, model architecture, optimizer, learning rate scheduler, and data loading are managed through JSON files located in the `main/configs/` directory. Each file (`criterion_config.json`, `dataloader_config.json`, `lr_config.json`, `model_config.json`, `opt_config.json`, `train_config.json`, `wandb_config.json`) defines specific parameters for its respective component.

To customize your training runs, modify these JSON files directly. The `main/train.py` script automatically loads these configurations at runtime.



### Inference & Quantization

The `inf_prof/` directory contains scripts and configurations for inference and quantization. Refer to `inf_prof/README.md` for detailed instructions on running inference sweeps and applying different quantization methods.

### Configuration & Profiling

The `conf_prof/` directory provides tools for profiling and hyperparameter sweeping related to training. To run a sweep and analyze results:

```bash
# From the project root directory
pip install -r conf_prof/requirements.txt # Ensure dependencies are installed
torchrun --standalone --nnodes=1 --nproc-per-node=<num_gpus> conf_prof/sweep.py
python conf_prof/analyze_times.py
```
Replace `<num_gpus>` with the number of GPUs you want to use for the sweep.

Results of the sweeps are stored in `conf_prof/results/`.
