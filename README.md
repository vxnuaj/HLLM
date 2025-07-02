# Hardware Aware SLM Pre-Training Infrastructure (Work in Progress)

This repository contains code for building, profiling, and training a variant of the LLaMA model architecture. It supports multiple attention mechanisms (e.g., Multihead Attention, Multi Query Attention, Grouped Query Attention, Multi-Value Attention), as well as dynamic NTK-Aware RoPE and includes scripts to analyze runtime performance across various configurations, with Distributed Infrastructure (DDP or FSDP).

### Structure

1. Under `model`, you'll find the definition for the model (LLaMA variants), and various attention mechanisms.
2. Under `main`, you'll find the main training scripts / utility training scripts
3. Under `conf_prof`, you'll find a set of scripts used for inference / backpropagation profiling.

### Usage

> ( has been tested on 2xA100s )

1. Define the Search Space by modifying `search_space.yaml`\
2. Connect to a cloud GPU instance (tested on 2xA100s)
3. Run:
    ```bash
    chmod +x conf_prof/sweep.sh
    ./conf_prof/sweep.sh 
    ```
4. Run:
    ```bash
    python conf_prof/analyze_times.py
    ```
5. View results under `conf_proj/results/results.json`
7. Preprocess & Train:
    ```bash
    cd main 
    chmod +x run.sh
    ./run.sh
    ```

# Training

**Compute**: 2x4090s

# TODO

### Pre-Trianing Run

- [X] `train_utils.py`
- [X] `train.py`
- [X] Figure out if there's anything else you can do to optimize the training loop.

## Training

- [ ] Figure out final configurations for the model ( .json files ).
  - [x] dataloader_config.json
  - [X] loss_config.json
  - [X] lr_config.json
  - [X] model_config.json
  - [X] opt_config.json
  - [X] train_config.json
    - [X] Make HF Repo for Model
    - [X] Update the train_config.json
  - [X] wandb_config.json
  - [X] Make HF Repo for the dataset and add the upload script for the dataset and the tokenizer.

- [X] Test Pipeline | on GPU.
  - [X] Tokenizer a small set of samples.
    -[X] Verify that we'll be using the 0th index for ignore_index in the loss.
  - [X] turn logger to logging on a single dist | or at option for it on parameters.
    - [X] DDP 
      - [X] Fix the all reduce
      - [X] Fix Progress Bar 
      - [X] Are epochs set right...?
    - [X] FSDP
  - [X] Verify model is learning
  - [X] Verify that the model is saving checkpoints locally.
  - [X] Verify that the checkpoints are properly saved (reusable).
  - [X] Make sure, given same exact configuration, we're starting off from the samep point in the dataset ( properly best to save a seed in the dataloader and input to torch.save )
  - [X] Verify that the model is saving logs locally.
  - [X] Verify that the model is uploading checkpoints to hugging face.
  - [X] Verify that the model is uploading dataset to hugging face.
  - [X] Verify that the model is logging to wandb.
  - [ ] dont want logs to interrupt progress bar, but want to be saved to the .log file.

- [ ] Prep for Pre-Training

  - [ ] Batch Size - Stress test VRAM to proper batch size for the dataloader_config.json
  - [ ] Figure out final hyperparameters for the LR Schedule based on batch size.

- [ ] Begin Training

### Inference Hyperparamter Sweep

- [ ] Build out Hyperparameter Sweep for Inference ( using existing model weights for primlinary runs but not actually using it for real results until we finish training the model )
  - [ ] TBD
  - [ ] TBD