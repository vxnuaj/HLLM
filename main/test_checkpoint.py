import torch

checkpoint = torch.load("main/checkpoints/RUN_001/RUN_001_DATETIME_2025-07-03_21-51-50_EPOCH_3_STEP_1353_GLOBAL_STEPS_5424.pt")

print(checkpoint['global_steps'])