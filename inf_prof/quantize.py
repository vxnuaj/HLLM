'''

TODO

- [X] gptq implementation
    - [ ] test / run quantization
- [ ] awq implementation
    - [ ] test / run quantization
- [ ] gguf implementation
    - [ ] test / run quantization
- [ ] exl12 implementation
    - [ ] test / run quantization
- [ ] bitsandbytes implementation
    - [ ] test / run quantization

'''


import os
import sys
import json
import logging
import torch
import torch.nn as nn
import pkl

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'package', 'gptq'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
from gptq import GPTQ
from quant import Quantizer
from model import Athena

logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_weights_model(model, weights_path, load_model_weights=True):
    model_weights = torch.load(weights_path, weights_only=True)
    if load_model_weights:
        model.load_state_dict(model_weights)
        return model, model_weights
    return model, model_weights

def find_layers(module, layers=[torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
 
def quantize_gptq(model, quantization_dtype=4, nsamples=128, calibration_data=None, 
             percdamp=0.01, groupsize=128):
    
    logger.info('Initializing GPTQ Quantization.')
    
    model.eval()
    wbits = quantization_dtype if isinstance(quantization_dtype, int) else 4
    
    model.embeddings = model.embeddings.cuda()
    if hasattr(model, 'pe'):
        model.pe = model.pe.cuda()
    model.block[0] = model.block[0].cuda()
            
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.context_len, model.d_model), dtype=dtype, device='cuda')
    cache = {'i': 0}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    
    model.block[0] = Catcher(model.block[0])
    
    if calibration_data is None:
        logger.warning("No calibration data provided, using random token sequences")
        calibration_data = [torch.randint(0, min(model.vocab_size, 32000), (1, model.context_len)) for _ in range(nsamples)]
    
    for i, batch in enumerate(calibration_data):
        if i >= nsamples:
            break
        try:
            if isinstance(batch, list):
                batch = batch[0]
            x = model.embeddings(batch.cuda())
            if hasattr(model, 'pe'):
                x = model.pe(x)
            model.block[0](x)
        except ValueError:
            pass
    
    model.block[0] = model.block[0].module
    
    model.block[0] = model.block[0].cpu()
    model.embeddings = model.embeddings.cpu()
    if hasattr(model, 'pe'):
        model.pe = model.pe.cpu()
    torch.cuda.empty_cache()
    
    logger.info('Calibration data collected. Starting quantization...')
    
    outs = torch.zeros_like(inps)
    quantizers = {}
    
    for i, layer in enumerate(model.block):
        layer = layer.cuda()
        subset = find_layers(layer)
        
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = Quantizer()
            gptq[name].quantizer.configure(wbits, perchannel=True, sym=False, mse=True)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for j in range(nsamples):
            if j < len(inps):
                outs[j] = layer(inps[j].unsqueeze(0), _inference=False)
        
        for h in handles:
            h.remove()
        
        for name in subset:
            logger.info(f'Quantizing block {i}, layer {name}')
            gptq[name].fasterquant(percdamp=percdamp, groupsize=groupsize)
            quantizers[f'block.{i}.{name}'] = gptq[name].quantizer
            gptq[name].free()
        
        layer = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    
    logger.info('GPTQ quantization complete')
    return model, quantizers

def quantize_awq(model):
    
    return model

def quantize(model, quantization_method, quant_save_path:str, model_id:str):
    
    '''
    TODO - verify before running quantization 
    ''' 
    
    quant_save_path = os.path.join(quant_save_path, model_id)
    if quantization_method == 'gptq':
        model, quantizers = quantize_gptq(model)
        torch.save({
            'model': model.state_dict(),
            'quantizers': quantizers,
            'model_id': model_id
            }, quant_save_path) 
    return