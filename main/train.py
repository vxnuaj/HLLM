import os
import sys 
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from train_utils import Trainer
from config import get_config

root_path = 'main/configs'

criterion_config = get_config(root_path = root_path, config_type = 'criterion')
lr_config = get_config(root_path = root_path, config_type = 'lr')
opt_config = get_config(root_path = root_path, config_type = 'opt')
train_config = get_config(root_path = root_path, config_type = 'train')
wandb_config = get_config(root_path = root_path, config_type = 'wandb')
dataloader_config = get_config(root_path = root_path, config_type = 'dataloader')
model_config = get_config(root_path = root_path, config_type = 'model')

parser = argparse.ArgumentParser()
parser.add_argument('--debug_nccl', action='store_true', help='Enable debug mode')
parser.add_argument('--debug_subsys', action = 'store_true', default = False, help='Debug subsystem')
parser.add_argument('--debug_level', type=str, default='INFO', help='Debug level')
args = parser.parse_args()

trainer = Trainer(
    model_config = model_config,
    criterion_config = criterion_config,
    dataloader_config = dataloader_config,
    optimizer_config = opt_config,
    scheduler_config = lr_config,
    wandb_config = wandb_config,
    train_config = train_config,
    debug_nccl = args.debug_nccl,
    debug_level = args.debug_level,
    debug_subsys = args.debug_subsys
)

trainer.train()
