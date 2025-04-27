These are the best top-10 configuration(s) across attn_type, with respect to avg_fwd_bwd_time.

# Top-10 configurations for mqa

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.010526601448655129`

Avg. Forward $\rightarrow$ Backward Time: `0.011017700247466565`

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
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.010526601448655129,
  "avg_fwd_bwd_time": 0.011017700247466565
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.011259399503469467`

Avg. Forward $\rightarrow$ Backward Time: `0.01123091921210289`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.011259399503469467,
  "avg_fwd_bwd_time": 0.01123091921210289
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.0508243615925312`

Avg. Forward $\rightarrow$ Backward Time: `0.011462128274142742`

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
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.0508243615925312,
  "avg_fwd_bwd_time": 0.011462128274142742
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.010980080924928188`

Avg. Forward $\rightarrow$ Backward Time: `0.011467584185302257`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.010980080924928188,
  "avg_fwd_bwd_time": 0.011467584185302257
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.010781869515776634`

Avg. Forward $\rightarrow$ Backward Time: `0.011623252332210541`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.010781869515776634,
  "avg_fwd_bwd_time": 0.011623252332210541
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.01162097103893757`

Avg. Forward $\rightarrow$ Backward Time: `0.011695174761116504`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.01162097103893757,
  "avg_fwd_bwd_time": 0.011695174761116504
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.011859335117042064`

Avg. Forward $\rightarrow$ Backward Time: `0.011754957064986228`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.011859335117042064,
  "avg_fwd_bwd_time": 0.011754957064986228
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.047648667767643926`

Avg. Forward $\rightarrow$ Backward Time: `0.011899131760001182`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.047648667767643926,
  "avg_fwd_bwd_time": 0.011899131760001182
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.006119761355221271`

Avg. Forward $\rightarrow$ Backward Time: `0.013493582680821419`

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
    "mixed_precision": true,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.006119761355221271,
  "avg_fwd_bwd_time": 0.013493582680821419
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.01747036002576351`

Avg. Forward $\rightarrow$ Backward Time: `0.019723271206021308`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": true,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.01747036002576351,
  "avg_fwd_bwd_time": 0.019723271206021308
}
```

---

# Top-10 configurations for mhsa

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.050280607864260675`

Avg. Forward $\rightarrow$ Backward Time: `0.010827900990843773`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.050280607864260675,
  "avg_fwd_bwd_time": 0.010827900990843773
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.01103117786347866`

Avg. Forward $\rightarrow$ Backward Time: `0.01128540761768818`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.01103117786347866,
  "avg_fwd_bwd_time": 0.01128540761768818
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.011285029239952565`

Avg. Forward $\rightarrow$ Backward Time: `0.011308038234710693`

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
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.011285029239952565,
  "avg_fwd_bwd_time": 0.011308038234710693
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.011475457325577736`

Avg. Forward $\rightarrow$ Backward Time: `0.011358139775693417`

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
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.011475457325577736,
  "avg_fwd_bwd_time": 0.011358139775693417
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.011483407840132713`

Avg. Forward $\rightarrow$ Backward Time: `0.011511997394263744`

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
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.011483407840132713,
  "avg_fwd_bwd_time": 0.011511997394263744
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.011749105714261531`

Avg. Forward $\rightarrow$ Backward Time: `0.011703617572784424`

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
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.011749105714261531,
  "avg_fwd_bwd_time": 0.011703617572784424
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.010455428324639797`

Avg. Forward $\rightarrow$ Backward Time: `0.011963996663689613`

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
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.010455428324639797,
  "avg_fwd_bwd_time": 0.011963996663689613
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.08887645527720452`

Avg. Forward $\rightarrow$ Backward Time: `0.012213894575834274`

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
    "flash_attn": false,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.08887645527720452,
  "avg_fwd_bwd_time": 0.012213894575834274
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.017007981054484846`

Avg. Forward $\rightarrow$ Backward Time: `0.019114704392850398`

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
    "mixed_precision": true,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.017007981054484846,
  "avg_fwd_bwd_time": 0.019114704392850398
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.017364067733287813`

Avg. Forward $\rightarrow$ Backward Time: `0.019293201714754106`

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
    "mixed_precision": true,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.017364067733287813,
  "avg_fwd_bwd_time": 0.019293201714754106
}
```

---

# Top-10 configurations for gqa

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.019569650329649448`

Avg. Forward $\rightarrow$ Backward Time: `0.01612737886607647`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.019569650329649448,
  "avg_fwd_bwd_time": 0.01612737886607647
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.017312996871769428`

Avg. Forward $\rightarrow$ Backward Time: `0.01726061772555113`

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
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.017312996871769428,
  "avg_fwd_bwd_time": 0.01726061772555113
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.01720694001764059`

Avg. Forward $\rightarrow$ Backward Time: `0.017468663044273854`

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
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.01720694001764059,
  "avg_fwd_bwd_time": 0.017468663044273854
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.018354738615453243`

Avg. Forward $\rightarrow$ Backward Time: `0.01880201853811741`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.018354738615453243,
  "avg_fwd_bwd_time": 0.01880201853811741
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02005308024585247`

Avg. Forward $\rightarrow$ Backward Time: `0.01945635795593262`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02005308024585247,
  "avg_fwd_bwd_time": 0.01945635795593262
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02205574493855238`

Avg. Forward $\rightarrow$ Backward Time: `0.021530226655304432`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02205574493855238,
  "avg_fwd_bwd_time": 0.021530226655304432
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.020659515485167503`

Avg. Forward $\rightarrow$ Backward Time: `0.023777065612375737`

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
    "mixed_precision": true,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.020659515485167503,
  "avg_fwd_bwd_time": 0.023777065612375737
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02156557723879814`

Avg. Forward $\rightarrow$ Backward Time: `0.02384819056838751`

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
    "mixed_precision": true,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02156557723879814,
  "avg_fwd_bwd_time": 0.02384819056838751
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.021559704542160035`

Avg. Forward $\rightarrow$ Backward Time: `0.0266808320581913`

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
    "mixed_precision": true,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.021559704542160035,
  "avg_fwd_bwd_time": 0.0266808320581913
}
```

---

Number of Groups: `2`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02576403383165598`

Avg. Forward $\rightarrow$ Backward Time: `0.02937695726752281`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": true,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02576403383165598,
  "avg_fwd_bwd_time": 0.02937695726752281
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.016976985819637774`

Avg. Forward $\rightarrow$ Backward Time: `0.017161251828074455`

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
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.016976985819637774,
  "avg_fwd_bwd_time": 0.017161251828074455
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.015611427091062068`

Avg. Forward $\rightarrow$ Backward Time: `0.017439540885388852`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.015611427091062068,
  "avg_fwd_bwd_time": 0.017439540885388852
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.017668162100017072`

Avg. Forward $\rightarrow$ Backward Time: `0.01859916027635336`

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
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.017668162100017072,
  "avg_fwd_bwd_time": 0.01859916027635336
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.019456355087459087`

Avg. Forward $\rightarrow$ Backward Time: `0.019437723718583585`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.019456355087459087,
  "avg_fwd_bwd_time": 0.019437723718583585
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.0221204949170351`

Avg. Forward $\rightarrow$ Backward Time: `0.022177136577665804`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.0221204949170351,
  "avg_fwd_bwd_time": 0.022177136577665804
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.021310035288333893`

Avg. Forward $\rightarrow$ Backward Time: `0.023482306711375713`

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
    "mixed_precision": true,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.021310035288333893,
  "avg_fwd_bwd_time": 0.023482306711375713
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.020314933061599733`

Avg. Forward $\rightarrow$ Backward Time: `0.0235363444685936`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.020314933061599733,
  "avg_fwd_bwd_time": 0.0235363444685936
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02110277961939573`

Avg. Forward $\rightarrow$ Backward Time: `0.023537164330482484`

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
    "mixed_precision": true,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02110277961939573,
  "avg_fwd_bwd_time": 0.023537164330482484
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.021221255920827388`

Avg. Forward $\rightarrow$ Backward Time: `0.026995364427566528`

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
    "mixed_precision": true,
    "flash_attn_dtype": "None",
    "compile": true,
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.021221255920827388,
  "avg_fwd_bwd_time": 0.026995364427566528
}
```

---

Number of Groups: `4`

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02547381293028593`

Avg. Forward $\rightarrow$ Backward Time: `0.029270396456122397`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": true,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02547381293028593,
  "avg_fwd_bwd_time": 0.029270396456122397
}
```

---

# Top-10 configurations for mva

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02062102161347866`

Avg. Forward $\rightarrow$ Backward Time: `0.02052180461585522`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02062102161347866,
  "avg_fwd_bwd_time": 0.02052180461585522
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.028251318596303462`

Avg. Forward $\rightarrow$ Backward Time: `0.02074215967208147`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": true,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.028251318596303462,
  "avg_fwd_bwd_time": 0.02074215967208147
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.01742274798452854`

Avg. Forward $\rightarrow$ Backward Time: `0.020765523426234722`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.01742274798452854,
  "avg_fwd_bwd_time": 0.020765523426234722
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02009157110005617`

Avg. Forward $\rightarrow$ Backward Time: `0.0208262412622571`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02009157110005617,
  "avg_fwd_bwd_time": 0.0208262412622571
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.020698220133781434`

Avg. Forward $\rightarrow$ Backward Time: `0.021206373497843743`

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
    "fsdp_wrap_policy": "transformer"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.020698220133781434,
  "avg_fwd_bwd_time": 0.021206373497843743
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02437472518533468`

Avg. Forward $\rightarrow$ Backward Time: `0.0214517106115818`

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
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02437472518533468,
  "avg_fwd_bwd_time": 0.0214517106115818
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.021152226217091084`

Avg. Forward $\rightarrow$ Backward Time: `0.021513385623693468`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "ddp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.021152226217091084,
  "avg_fwd_bwd_time": 0.021513385623693468
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.02205592505633831`

Avg. Forward $\rightarrow$ Backward Time: `0.02232541985809803`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.02205592505633831,
  "avg_fwd_bwd_time": 0.02232541985809803
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.022937316931784153`

Avg. Forward $\rightarrow$ Backward Time: `0.022405436784029006`

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
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.022937316931784153,
  "avg_fwd_bwd_time": 0.022405436784029006
}
```

---

Avg. Forward Time: `0.0`

Avg. Backward Time: `0.021404006108641623`

Avg. Forward $\rightarrow$ Backward Time: `0.022595212087035178`

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
    "flash_attn": true,
    "pos_emb_type": "rope",
    "mixed_precision": false,
    "flash_attn_dtype": "torch.float16",
    "compile": true,
    "parallel": "fsdp",
    "fsdp_wrap_policy": "auto"
  },
  "avg_forward_time": 0.0,
  "avg_backward_time": 0.021404006108641623,
  "avg_fwd_bwd_time": 0.022595212087035178
}
```

---

