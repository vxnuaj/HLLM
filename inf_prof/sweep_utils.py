import os
import sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from model import LLaMA

def load_weights(model, weights_path):
   
    torch.load(model_weights, )
    
    return