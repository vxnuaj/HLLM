#!/bin/bash

pip install -r requirements.txt
torchrun --standalone --nnodes=1 --nproc-per-node=2 conf_prof/sweep.py
python conf_prof/analyze_times.py