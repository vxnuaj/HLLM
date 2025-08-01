'''
the `run_profs` function from `sweep_utils.py` with predefined
configuration parameters to measure and analyze the performance of different
model configurations.
'''

run_profs(
    cfg_path="conf_prof/search_space.yaml",
    data_shape=(16, 512), 
    vocab_size = 10000,
    n_inf_passes=50,
    n_bck_passes=50,
    n_fwd_bck_iter=50,
    results_root = 'conf_prof/results',
    profile_forward = False,
    compile_warmup_steps = 7 
)