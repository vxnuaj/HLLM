# Hardware Aware SLM Pre-Training Infrastructure (Work in Progress)

This repository contains code for building, profiling, and training a variant of the LLaMA model architecture. It supports multiple attention mechanisms (e.g., Multihead Attention, Multi Query Attention, Grouped Query Attention, Multi-Value Attention) and includes scripts to analyze runtime performance across various configurations, with Distributed Infrastructure (DDP or FSDP)

### Usage

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
