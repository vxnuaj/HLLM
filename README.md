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


### TODO

- [ ] Final Inference & Automated FLOPs analysis.
- [ ] Stress Test.
