import os
import sys 

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

trainer = Trainer(
    model_config = model_config,
    criterion_config = criterion_config,
    dataloader_config = dataloader_config,
    optimizer_config = opt_config,
    scheduler_config = lr_config,
    wandb_config = wandb_config,
    train_config = train_config
)

trainer.train()
