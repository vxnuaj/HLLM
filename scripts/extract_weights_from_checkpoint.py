##### This script extracts the model weights from a given checkpoint and uploads to hf.

'''
python extract_weights_from_checkpoint.py --checkpoint_root_path main/checkpoints \
                                        --run_id 001 \
                                        --file_name {FILE_NAME} \
                                        --hf_repo_id tiny-research/athena \
                                        --hf_path ATHENA_V1_TINY_39.7M/final_model_parameters 
'''

import torch
import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login as hf_login

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_root_path", type=str, required=True, help="Path to the checkpoint file")
parser.add_argument("--run_id", type=str, required=True, help="Run ID of the checkpoint")
parser.add_argument("--file_name", type=str, required=True, help="File name of the checkpoint")
parser.add_argument("--hf_repo_id", type=str, required=True, help="Model ID on Hugging Face ( folder within the repo )")
parser.add_argument("--hf_path", type=str, required=True, help="Model ID on Hugging Face ( folder within the repo )")
args = parser.parse_args()

checkpoint = torch.load(os.path.join(f"{args.checkpoint_root_path}/RUN_{args.run_id}", args.file_name))
model_state_dict = checkpoint['model_state_dict']

print("Logging in to Hugging Face with token: ", os.getenv('HF_TOKEN'))
hf_login(token = os.getenv('HF_TOKEN'))   
api = HfApi() 

api.upload_file(
    path_or_fileobj = model_state_dict,
    path_in_repo = args.hf_path,
    repo_id = args.hf_repo_id,
    repo_type = 'model',
)