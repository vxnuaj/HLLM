"""This script handles the preprocessing of TinyStories data, including downloading, tokenization,
creating input-target sequences, and optionally uploading the processed data and tokenizer
to the Hugging Face Hub.

It performs the following main steps:
1.  Downloads TinyStories datasets.
2.  Processes raw text data.
3.  Trains a new BPE tokenizer or loads an existing one.
4.  Tokenizes the processed data.
5.  Generates input (X) and target (y) sequences for training and validation,
    saving them as PyTorch `.pt` files.
6.  Optionally uploads the tokenizer and processed data to a Hugging Face dataset repository.

Key parameters are defined at the top of the script, controlling aspects like
vocabulary size, context length, number of processes, and output paths.
"""
import torch
import os
import tinytok.core as tt
import argparse

from datetime import datetime
from huggingface_hub import HfApi, login, whoami
from dotenv import load_dotenv

load_dotenv(dotenv_path = 'main/configs/.env')

hf_token = os.environ.get("HF_TOKEN")

if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is not set. \
                     Please set it in your .env file.")

login(token=hf_token)  

print(f"Logged in as: {whoami()['name']}")


os.environ["TOKENIZERS_PARALLELISM"] = "false"

save_dir = 'data/'
file_1 = 'data/train1.parquet'
file_2 = 'data/train2.parquet'
file_3 = 'data/train3.parquet'
file_4 = 'data/train4.parquet'
file_val = 'data/validation.parquet'

file_train = [file_1, file_2, file_3, file_4]
file_val = [file_val]

# PARAMS -----------------

return_single_str = False
vocab_size = 4_906 #
special_tokens = {'pad': '<|pad|>','bos': '<|bos|>', 'eos': '<|eos|>', } # toks (0, 1) respectively
save_tokenizer_path = 'data/tokenizer.json'
context_len = 512
processes = 24
flat_tensor = True
flat_tensor_val = False
seq_tensor_size = 25_000
val_seq_tensor_size = None
max_toks = 266_666_666 # per chinchilla scaling laws
val_max_toks = None
batch_first = True
val_train_n_samples = 2000

if __name__ == "__main__":


    X_train_pth = 'data/tensors/train/X'
    y_train_pth = 'data/tensors/train/y'
    X_val_pth = 'data/tensors/val/X'
    y_val_pth = 'data/tensors/val/y'
    
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sample", action="store_true")
    argparser.add_argument("--upload_hf", action="store_true")
    args = argparser.parse_args()
 
    tt.download_tinystories(save_dir) 
  
    os.makedirs(X_train_pth, exist_ok = True)
    os.makedirs(y_train_pth, exist_ok = True)
    os.makedirs(X_val_pth, exist_ok = True)
    os.makedirs(y_val_pth, exist_ok = True)
    os.makedirs(save_dir, exist_ok = True)

    # --- processing ---
    
    data = tt.data_process(
        files=file_train,
        bos_str=special_tokens['bos'],
        eos_str=special_tokens['eos'],
        return_single_str=return_single_str,
        processes=processes
    )

    tokenizer = tt.train_new_tokenizer_bpe(
        data=data['text'].tolist(),
        vocab_size=vocab_size,
        special_tokens=list(special_tokens.values()),
        save_path=save_tokenizer_path
    )
    
    data_tensor = tt.tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor=flat_tensor,
        processes=processes
    )

    if isinstance(seq_tensor_size, int):
        sequence_generator = tt.create_train_sequences_gen(
            data=data_tensor,
            context_len=context_len,
            seq_tensor_size=seq_tensor_size,
            max_toks=max_toks,
            processes=processes
        )
        for i, (X, y) in enumerate(sequence_generator):
            
            torch.save(X, os.path.join(X_train_pth, f'X_train_{i}.pt'))
            torch.save(y, os.path.join(y_train_pth, f'y_train_{i}.pt'))
           
    elif not seq_tensor_size:

        X_train, y_train = tt.create_train_sequences_gen(
            data=data_tensor,
            context_len=context_len,
            seq_tensor_size=seq_tensor_size,
            max_toks=max_toks,
            processes=processes
        )

        torch.save(X_train, os.path.join(X_train_pth, "X_train.pt"))
        torch.save(y_train, os.path.join(y_train_pth, "y_train.pt"))

        del X_train, y_train

    # ---  validation ---

    data = tt.data_process(
        files=file_val,
        bos_str=special_tokens['bos'],
        eos_str=special_tokens['eos'],
        return_single_str=return_single_str,
        processes=processes
    )

    data_tensor = tt.tokenize(
        data=data,
        tokenizer=tokenizer,
        flat_tensor=flat_tensor_val,
        processes=processes,
    )

    X_val, Y_val = tt.create_val_sequences(
        data=data_tensor,
        batch_first=batch_first,
        padding_value=tokenizer.encode(special_tokens['pad']).ids[0]
    )
   
    X_val_subset = X_val[0:val_train_n_samples+1]
    Y_val_subset = Y_val[0:val_train_n_samples+1]
    
    torch.save(X_val_subset, os.path.join(X_val_pth, 'X_val.pt'))
    torch.save(Y_val_subset, os.path.join(y_val_pth, 'Y_val.pt'))
   
    # --- upload to hugging face ---

    if args.upload_hf:

        X_train_toks, _, _ = tt.get_token_count(X_train_pth, y_train_pth, return_total_toks = True)
        X_val_toks, _, _ = tt.get_token_count(X_val_pth, y_val_pth, return_total_toks = True)

        total_token_count = X_train_toks + X_val_toks
    
        if args.sample:
            X_path_in_repo = f'CONTEXT_LEN_{context_len}_TOKENS_{total_token_count}_SAMPLE/TRAIN_TOKENS_{X_train_toks}_TIME_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
        else:
            X_path_in_repo = f'CONTEXT_LEN_{context_len}_TOKENS_{total_token_count}/TRAIN_TOKENS_{X_train_toks}_TIME_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
    
        if args.sample:
            Y_path_in_repo = f'CONTEXT_LEN_{context_len}_TOKENS_{total_token_count}_SAMPLE/TRAIN_TOKENS_{X_train_toks}_TIME_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
        else:
            Y_path_in_repo = f'CONTEXT_LEN_{context_len}_TOKENS_{total_token_count}/TRAIN_TOKENS_{X_train_toks}_TIME_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
    
        
        api = HfApi()

        api.create_repo(
            repo_id="tiny-research/TinyStories",
            repo_type="dataset",
            exist_ok=True
        )
        
        print("Uploading Tokenizer.") 
        
        api.upload_file( # the tokenizer
            path_or_fileobj=save_tokenizer_path,
            repo_id='tiny-research/TinyStories',
            path_in_repo=f'CONTEXT_LEN_{context_len}_TOKENS_{total_token_count}/tokenizer.json',
            repo_type = 'dataset'
        )

        print('Uploading Training Files.')

        api.upload_folder(
            folder_path=X_train_pth,
            repo_id='tiny-research/TinyStories',
            path_in_repo=f'{X_path_in_repo}',
            repo_type = "dataset"
        )

        api.upload_folder(
            folder_path = y_train_pth,
            repo_id='tiny-research/TinyStories',
            path_in_repo=f'{Y_path_in_repo}',
            repo_type = "dataset"        
        )

        print('Uploading Validation Files.')

        api.upload_folder(
            folder_path=X_val_pth,
            repo_id='tiny-research/TinyStories',
            path_in_repo=f'{X_path_in_repo}',
            repo_type = "dataset"        
        )

        api.upload_folder(
            folder_path=y_val_pth,
            repo_id='tiny-research/TinyStories',
            path_in_repo=f'{Y_path_in_repo}',
            repo_type = "dataset"        
        )