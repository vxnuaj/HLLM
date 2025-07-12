import wandb
api = wandb.Api()

run = api.run("vxnuaj/ATHENA/ATHENA_V1_TINY_39.7M_RUN_001")

print(run.files())