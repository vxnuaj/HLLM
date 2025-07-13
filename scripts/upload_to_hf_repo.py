import os

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

load_dotenv(dotenv_path = 'main/configs/.env')
login(token = os.getenv('HF_TOKEN'))

api = HfApi()

api.upload_file(
    path_or_fileobj = "main/checkpoints/RUN_002/RUN_002_DATETIME_2025-07-12_21-36-20_EPOCH_5_STEP_1357_GLOBAL_STEPS_6785.pt",
    path_in_repo = "ATHENA_V1_TINY_39.7M/checkpoints/RUN_002/RUN_002_DATETIME_2025-07-12_21-36-20_EPOCH_5_STEP_1357_GLOBAL_STEPS_6785.pt", 
    repo_id = "tiny-research/athena",
    repo_type = "model",
    )