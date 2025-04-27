These are the best top-1 configuration(s) across attn_type, with respect to avg_fwd_bwd_time.

# Top-1 configurations for gqa

Number of Groups: `2`

Avg. Forward Time: `0.004043317027390003`

Avg. Backward Time: `0.019750128965824842`

Avg. Forward $\rightarrow$ Backward Time: `0.020704832673072816`

```json
{
  "config": {
    "context_len": 512,
    "d_model": 512,
    "n_heads": 8,
    "n_blocks": 4,
    "vocab_size": 10000,
    "pos_emb_dropout_p": 0.1,
    "learned": false,
    "ntk_rope_scaling": false,
    "dyn_scaling": false,
    "attn_type": "gqa",
    "n_groups": 2,
    "top_k_sparsev": null,
    "p_threshold": null,
    "p_threshold_steps_fraction": null,
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.004043317027390003,
  "avg_backward_time": 0.019750128965824842,
  "avg_fwd_bwd_time": 0.020704832673072816
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.003723383042961359`

Avg. Backward Time: `0.020177032966166734`

Avg. Forward $\rightarrow$ Backward Time: `0.02050905155017972`

```json
{
  "config": {
    "context_len": 512,
    "d_model": 512,
    "n_heads": 8,
    "n_blocks": 4,
    "vocab_size": 10000,
    "pos_emb_dropout_p": 0.1,
    "learned": false,
    "ntk_rope_scaling": false,
    "dyn_scaling": false,
    "attn_type": "gqa",
    "n_groups": 4,
    "top_k_sparsev": null,
    "p_threshold": null,
    "p_threshold_steps_fraction": null,
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.003723383042961359,
  "avg_backward_time": 0.020177032966166734,
  "avg_fwd_bwd_time": 0.02050905155017972
}
```

---

# Top-1 configurations for mhsa

Avg. Forward Time: `0.0017296219989657402`

Avg. Backward Time: `0.01115017794072628`

Avg. Forward $\rightarrow$ Backward Time: `0.010196861457079648`

```json
{
  "config": {
    "context_len": 512,
    "d_model": 512,
    "n_heads": 8,
    "n_blocks": 4,
    "vocab_size": 10000,
    "pos_emb_dropout_p": 0.1,
    "learned": false,
    "ntk_rope_scaling": false,
    "dyn_scaling": false,
    "attn_type": "mhsa",
    "n_groups": null,
    "top_k_sparsev": null,
    "p_threshold": null,
    "p_threshold_steps_fraction": null,
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0017296219989657402,
  "avg_backward_time": 0.01115017794072628,
  "avg_fwd_bwd_time": 0.010196861457079648
}
```

---

# Top-1 configurations for mva

Avg. Forward Time: `0.004675361402332783`

Avg. Backward Time: `0.022196939047425986`

Avg. Forward $\rightarrow$ Backward Time: `0.018242416847497226`

```json
{
  "config": {
    "context_len": 512,
    "d_model": 512,
    "n_heads": 8,
    "n_blocks": 4,
    "vocab_size": 10000,
    "pos_emb_dropout_p": 0.1,
    "learned": false,
    "ntk_rope_scaling": false,
    "dyn_scaling": false,
    "attn_type": "mva",
    "n_groups": null,
    "top_k_sparsev": null,
    "p_threshold": null,
    "p_threshold_steps_fraction": null,
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.004675361402332783,
  "avg_backward_time": 0.022196939047425986,
  "avg_fwd_bwd_time": 0.018242416847497226
}
```

---

# Top-1 configurations for mqa

Avg. Forward Time: `0.0018788027577102185`

Avg. Backward Time: `0.010332888402044773`

Avg. Forward $\rightarrow$ Backward Time: `0.010391169507056475`

```json
{
  "config": {
    "context_len": 512,
    "d_model": 512,
    "n_heads": 8,
    "n_blocks": 4,
    "vocab_size": 10000,
    "pos_emb_dropout_p": 0.1,
    "learned": false,
    "ntk_rope_scaling": false,
    "dyn_scaling": false,
    "attn_type": "mqa",
    "n_groups": null,
    "top_k_sparsev": null,
    "p_threshold": null,
    "p_threshold_steps_fraction": null,
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0018788027577102185,
  "avg_backward_time": 0.010332888402044773,
  "avg_fwd_bwd_time": 0.010391169507056475
}
```

---

