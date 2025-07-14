def generate(
    str_in:str, 
    tokenizer, 
    model, 
    eos_token, 
    context_len:int,
    max_toks_out = None, 
    _greedy:bool = False, 
    top_p:float = .5, 
    top_k:int = None, 
    temperature:float = 1.0,
    verbose:bool = False,
    new_line:bool = False,
    *args,
    **kwargs
    ):
  
    model.eval()
    
    if new_line:
        print()
    sampler = sample_model(
        str_in = str_in,
        tokenizer = tokenizer,
        model = model,
        context_len = context_len,
        eos_token=eos_token,
        max_toks_out = max_toks_out,
        _greedy = _greedy,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature
    ) 
    n_toks = 0
    if verbose:
        start_time = time.time()
    for i in sampler:
        n_toks += 1 
        print(i, end = '', flush = True)
    if verbose: 
        end_time = time.time()
        print(f'\n\nGenerated a total of {n_toks} tokens in {end_time - start_time} seconds')
    if new_line:
        print()

    model.t = None
    
    for block in model.transformer_blocks:
        block.MHSA.K_cache = None
        block.MHSA.V_cache = None
