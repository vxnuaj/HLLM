import torch
import numpy as np
import os 
import time
import json
from pathlib import Path
from pprint import pprint
from torchinfo import summary

from model import Athena

# Load model configuration from JSON
config_path = Path(__file__).parent.parent / 'main' / 'configs' / 'model_config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

print("\n=== Model Configuration ===")
pprint(config)
print("=" * 50 + "\n")

batch_size = 1
seq_len = 512

context_len = config['context_len']
d_model = config['d_model']
n_heads = config['n_heads']
n_blocks = config['n_blocks']
vocab_size = config['vocab_size']
pos_emb_dropout_p = config['pos_emb_dropout_p']
pos_emb_type = config['pos_emb_type']
learned = config['learned']
ntk_rope_scaling = config['ntk_rope_scaling']
dyn_scaling = config['dyn_scaling']
attn_type = config['attn_type']
n_groups = config['n_groups']
top_k_sparsev = config['top_k_sparsev']
p_threshold = config['p_threshold']
p_threshold_steps_fraction = config['p_threshold_steps_fraction']
supress_warnings = config['supress_warnings']
flash_attn = config['flash_attn']
flash_attn_dtype = config['flash_attn_dtype']
model_name = config['model_name']
model_series_name = config['model_series_name']

# Create model with config
model = Athena(
    context_len=context_len,
    d_model=d_model,
    n_heads=n_heads,
    n_blocks=n_blocks,
    vocab_size=vocab_size,
    pos_emb_dropout_p=pos_emb_dropout_p,
    pos_emb_type=pos_emb_type,
    learned=learned,
    ntk_rope_scaling=ntk_rope_scaling,
    dyn_scaling=dyn_scaling,
    attn_type=attn_type,
    n_groups=n_groups,
    top_k_sparsev=top_k_sparsev,
    p_threshold=p_threshold,
    p_threshold_steps_fraction=p_threshold_steps_fraction,
    supress_warnings=supress_warnings,
    flash_attn=flash_attn,
    flash_attn_dtype=flash_attn_dtype,
    model_name=model_name,
    model_series_name=model_series_name
)

x = torch.randint(low = 0, high = vocab_size, size = (batch_size, seq_len))

summary(model, input_data=x)
