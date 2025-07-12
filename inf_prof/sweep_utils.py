import os
import sys
import yaml
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from model import LLaMA

def load_weights_model(model, weights_path, load_model_weights = True):
    model_weights = torch.load(weights_path, weights_only = True )
    if load_model_weights:
        model.load_state_dict(model_weights)
        return model, model_weights
   
def quantization_or_amp(model):
    
    return  
  
   
def main(
    model,
    weights_path:str,
    load_model_weights:bool = True,
    ):
    
    model, _ = load_weights_model(model, weights_path = weights_path, load_model_weights = load_model_weights)


    
    return