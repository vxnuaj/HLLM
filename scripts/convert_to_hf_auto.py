"""
Script to convert custom Athena model to HuggingFace AutoModelForCausalLM format.
"""

import os
import sys
import argparse
import torch

from model import *  
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'main'))

from model import Athena  
from configuration_athena import AthenaConfig
from modeling_athena import AthenaForCausalLM
from config import get_config

load_dotenv()
if os.environ.get('HF_TOKEN'):
    login(token=os.environ['HF_TOKEN'])

def convert_model_weights(original_model: Athena, hf_model: AthenaForCausalLM):
    """
    Convert weights from original model to HuggingFace model format.
    
    Args:
        original_model (Athena): The original trained model
        hf_model (AthenaForCausalLM): The HuggingFace-compatible model
    """
    print("Converting model weights...")
    
    original_state_dict = original_model.state_dict()
    hf_state_dict = hf_model.state_dict()
    
    
    weight_mapping = {}
    
    
    if "embeddings.weight" in original_state_dict:
        weight_mapping["embeddings.weight"] = "embeddings.weight"
    
    
    for i in range(original_model.n_blocks):
        orig_prefix = f"block.{i}"
        hf_prefix = f"block.{i}"
        
        
        attention_mappings = {
            f"{orig_prefix}.attn.linearQ.weight": f"{hf_prefix}.attn.linearQ.weight",
            f"{orig_prefix}.attn.linearK.weight": f"{hf_prefix}.attn.linearK.weight",
            f"{orig_prefix}.attn.linearV.weight": f"{hf_prefix}.attn.linearV.weight",
            f"{orig_prefix}.attn.linear_out.weight": f"{hf_prefix}.attn.linear_out.weight",
        }
        
        
        ff_mappings = {
            f"{orig_prefix}.swiglu.swiglu_linear.weight": f"{hf_prefix}.swiglu.swiglu_linear.weight",
            f"{orig_prefix}.swiglu.swiglu_gate_linear.weight": f"{hf_prefix}.swiglu.swiglu_gate_linear.weight",
            f"{orig_prefix}.swiglu.linear_out.weight": f"{hf_prefix}.swiglu.linear_out.weight",
        }
        
        
        norm_mappings = {
            f"{orig_prefix}.rmsnorm1.weight": f"{hf_prefix}.rmsnorm1.weight",
            f"{orig_prefix}.rmsnorm2.weight": f"{hf_prefix}.rmsnorm2.weight",
        }
        
        weight_mapping.update(attention_mappings)
        weight_mapping.update(ff_mappings)
        weight_mapping.update(norm_mappings)
    
    
    final_mappings = {
        "rmsnorm.weight": "rmsnorm.weight",
        "linear.weight": "linear.weight",
    }
    weight_mapping.update(final_mappings)
    
    
    converted_weights = {}
    missing_keys = []
    
    for original_key, hf_key in weight_mapping.items():
        if original_key in original_state_dict and hf_key in hf_state_dict:
            orig_weight = original_state_dict[original_key]
            expected_shape = hf_state_dict[hf_key].shape
            
            if orig_weight.shape == expected_shape:
                converted_weights[hf_key] = orig_weight
                print(f"✓ Mapped {original_key} -> {hf_key}")
            else:
                print(f"✗ Shape mismatch for {original_key}: {orig_weight.shape} vs {expected_shape}")
                missing_keys.append(original_key)
        else:
            missing_keys.append(original_key)
            print(f"✗ Could not find mapping for {original_key}")
    
    
    for key in hf_state_dict.keys():
        if key not in converted_weights:
            print(f"⚠ HF model key not mapped: {key}")
    
    
    hf_model.load_state_dict(converted_weights, strict=False)
    
    if missing_keys:
        print(f"\n⚠ Warning: {len(missing_keys)} keys could not be mapped:")
        for key in missing_keys[:5]:  
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")
    
    print("Weight conversion completed!")

def main():
    parser = argparse.ArgumentParser(description='Convert custom Athena model to HuggingFace AutoModelForCausalLM format')
    parser.add_argument('--config_root_path', type=str, required=True, help='Root path to model configs')
    parser.add_argument('--model_weights_path', type=str, required=True, help='Path to trained model weights (.pth file)')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the converted HuggingFace model')
    parser.add_argument('--hf_model_repo', type=str, help='HuggingFace model repository to push to (optional)')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--test_generation', action='store_true', help='Test the converted model with generation')
    
    args = parser.parse_args()
    
    print("Loading original model configuration...")
    model_config = get_config(root_path=args.config_root_path, config_type='model')
    
    print("Creating HuggingFace configuration...")
    hf_config = AthenaConfig(**model_config)
    
    print("Loading original model with trained weights...")
    original_model = Athena(**model_config)
    
    if args.model_weights_path:
        checkpoint = torch.load(args.model_weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            original_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            original_model.load_state_dict(checkpoint)
    
    print("Creating HuggingFace-compatible model...")
    hf_model = AthenaForCausalLM(hf_config)
    
    convert_model_weights(original_model, hf_model)
    
    print(f"Saving converted model to {args.save_path}...")
    os.makedirs(args.save_path, exist_ok=True)
    
    hf_model.save_pretrained(args.save_path)
    hf_config.save_pretrained(args.save_path)
    
    print("Testing model loading with AutoModelForCausalLM...")
    try:
        
        loaded_model = AutoModelForCausalLM.from_pretrained(args.save_path, trust_remote_code=True)
        loaded_config = AutoConfig.from_pretrained(args.save_path, trust_remote_code=True)
        
        print("Successfully loaded model with AutoModelForCausalLM!")
        print(f"Model type: {type(loaded_model)}")
        print(f"Config type: {type(loaded_config)}")
        
        test_input = torch.randint(0, loaded_config.vocab_size, (1, 10))
        with torch.no_grad():
            outputs = loaded_model(test_input)
        print(f"Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"Error testing model loading: {e}")
        return
    
    if args.test_generation:
        print("Testing text generation...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            model = loaded_model
            model.eval()
            
            test_text = "Once upon a time"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated text: {generated_text}")
            
        except Exception as e:
            print(f"Error during generation test: {e}")
    
    if args.hf_model_repo:
        print(f"Pushing model to HuggingFace Hub: {args.hf_model_repo}")
        try:
            hf_model.push_to_hub(args.hf_model_repo)
            hf_config.push_to_hub(args.hf_model_repo)
            print("Successfully pushed to HuggingFace Hub!")
        except Exception as e:
            print(f"Error pushing to Hub: {e}")
    
    print("\n" + "="*50)
    print("CONVERSION COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Your model is now saved at: {args.save_path}")
    print("You can now use it with:")
    print("  from transformers import AutoModelForCausalLM")
    print("  model = AutoModelForCausalLM.from_pretrained('{args.save_path}', trust_remote_code=True)")

if __name__ == "__main__":
    main()