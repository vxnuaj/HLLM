import torch.optim as opt
import torch.nn as nn
import os
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import LLaMA
from train_utils import Trainer, TrainerConfig, get_scheduler
from dataloader import get_data, get_dataloader

root_path = 'main/configs'
data_root_path = 'data/train'

loss_config = TrainerConfig.get_config(
    root_path = root_path,
    config_type = 'loss'
)

lr_config = TrainerConfig.get_config(
    root_path = root_path,
    config_type = 'lr'
)

opt_config = TrainerConfig.get_config(
    root_path = root_path,
    config_type = 'opt'
)

train_config = TrainerConfig.get_config(
    root_path = root_path,
    config_type = 'train'
)

wandb_config = TrainerConfig.get_config(
    root_path = root_path,
    config_type = 'run'
)

dataloader_config = TrainerConfig.get_config(
    root_path = root_path,
    config_type = 'dataloader'
)

model_config = TrainerConfig.get_config(
    root_path = root_path,
    config_type = 'model'
)

# DATA & DATALOADER

X, y = get_data(dataloader_config['data_root_path'])
dataloader = get_dataloader(X, y, **dataloader_config)
model = LLaMA(**model_config)
trainer_config = TrainerConfig(**train_config)
optimizer = opt.AdamW(**opt_config)
scheduler = get_scheduler(optimizer, **lr_config)
criterion = nn.CrossEntropy(**loss_config)

trainer = Trainer(
    model = model,
    criterion = criterion,
    dataloader = dataloader,
    scheduler = scheduler,
    config = trainer_config
)

trainer.train()