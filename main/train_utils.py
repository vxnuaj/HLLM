import torch
import torch.nn as nn
import torch.optim as opt
import torch.distributed as dist

import logging
import json
import math
import time
import os
import gc
import sys
import functools
import traceback
import wandb

from dataclasses import asdict
from dataloader import get_data
from config import ModelConfig, CriterionConfig, DataloaderConfig, OptimizerConfig, SchedulerConfig, TrainingConfig, WandbConfig
from torch.distributed.fsdp import StateDictType
from torch.distributed import ReduceOp
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from dataloader import get_dataloader
from contextlib import contextmanager
from datetime import datetime
from tqdm import tqdm
from rich.logging import RichHandler
from huggingface_hub import HfApi, create_repo, login as hf_login

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))

from model import LLaMA
from blocks import TransformerBlock
from dotenv import load_dotenv

class Trainer:
    def __init__(
        self,
        model_config:dict,
        criterion_config:dict,
        dataloader_config:dict,
        optimizer_config:dict,
        scheduler_config:dict,
        wandb_config:dict,
        train_config:dict 
        ):
    
        assert torch.cuda.is_available(), ValueError("No CUDA device found")
        torch.cuda.set_device(f"cuda:{os.environ.get('LOCAL_RANK')}")
      
        self.model_config = ModelConfig(**model_config)
        self.criterion_config = CriterionConfig(**criterion_config)
        self.optimizer_config = OptimizerConfig(**optimizer_config)
        self.lf_config = SchedulerConfig(**scheduler_config)
        self.dataloader_config = DataloaderConfig(**dataloader_config)
        self.wandb_config = WandbConfig(**wandb_config)
        self.train_config = TrainingConfig(**train_config)
      
        self.train_dataloader_config = self.dataloader_config.train_dataloader_config 
        self.val_dataloader_config = self.dataloader_config.val_dataloader_config 
       
        self.vocab_size = self.train_config.vocab_size 
        self.batch_size = self.train_dataloader_config['train_batch_size'] 
        self.context_length = self.train_config.context_length 
        self.epochs = self.train_config.epochs
        self.checkpoint_steps = self.train_config.checkpoint_steps
        self.save_checkpoint_path = self.train_config.save_checkpoint_path
        self.save_hf = self.train_config.save_hf
        self.hf_repo_id = self.train_config.hf_repo_config['hf_repo_id']
        self.hf_repo_exists = self.train_config.hf_repo_config['hf_repo_exists']
        self.hf_repo_type = self.train_config.hf_repo_config['hf_repo_type']
        self.hf_root_path = self.train_config.hf_repo_config['hf_root_path']
        
        self.val_steps = self.train_config.val_steps
        self.mixed_precision = self.train_config.mixed_precision
        self.mixed_precision_dtype = self._get_mixed_precision_dtype(self.train_config.mixed_precision_dtype) if self.train_config.mixed_precision else None
        self.max_grad_norm = self.train_config.max_grad_norm
        self.track_grad_norm = self.train_config.track_grad_norm
        self.parallel_type = self.train_config.parallel_type
        self.val_batch_size = self.val_dataloader_config['val_batch_size']
        self.val_num_workers = self.val_dataloader_config['val_num_workers']
        self.val_shuffle = self.val_dataloader_config['val_shuffle']
        self.val_pin_memory = self.val_dataloader_config['val_pin_memory']
        self.val_mixed_precision = self.train_config.val_mixed_precision
        self.val_mixed_precision_dtype = self._get_mixed_precision_dtype(self.train_config.val_mixed_precision_dtype) if self.train_config.val_mixed_precision else None
        self.val_data_root_path = self.val_dataloader_config['val_data_root_path']
        self.wandb_ = self.train_config.wandb 
        self._compile = self.train_config._compile
        self._compile_warmup_steps = self.train_config._compile_warmup_steps
        self.log_level = self.train_config.log_level
        self.log_root_path = self.train_config.log_root_path if hasattr(self.train_config, 'log_root_path') else None
        self.fsdp_wrap_policy = self.train_config.fsdp_wrap_policy
        self.model_name = self.model_config.model_name

        self.disable = self.train_config.disable
        self.disable_exclude = self.train_config.disable_exclude

        self.logger = logging.getLogger(__name__)      
        date_time = self.setup_logger(log_level=self.log_level, log_root_path=self.log_root_path, return_date_time = True)
        self.run_start_date_time = date_time
        
        load_dotenv(dotenv_path = 'main/configs/.env') 
        
        self.hf_token = os.environ.get('HF_TOKEN')
        self.wandb_token = os.environ.get('WANDB_API_KEY')
       
        assert self.hf_token is not None and self.save_hf, ValueError(f'HF_TOKEN must be specified if save_hf is True to save model \
                                                                      checkpoints to hugging face repo {self.hf_repo_id}')
        assert self.wandb_token is not None and self.wandb_, ValueError(f'WANDB_TOKEN must be specified if wandb is True, \
                                                            to log the training run to wandb project {self.wandb_config.project}')
      
       
        self.scaler = GradScaler() if self.mixed_precision else None

        self._setup_parallel()
        self.device = self._get_device()
        
        self.model = self._get_model(asdict(self.model_config))

        if dist.get_rank() == 0:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude):
                self.logger.info('SUCCESFULLY WRAPPED MODEL IN FSDP /DDP')

        sys.exit(0)
 
        self.criterion = nn.CrossEntropyLoss(**self.criterion_config)
        self.optimizer = opt.AdamW(self.model.parameters(), **asdict(self.optimizer_config))
        self.scheduler = get_scheduler(self.optimizer, **asdict(self.scheduler_config))
     
     
        if dist.get_rank() == 0:
            self.run_id = input("Enter Run ID (3 digit integer, e.g. 001):") 
        
        if self.wandb_:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude):
                self.logger.info(f"Logging to wandb project {self.wandb_config.project} as {self.wandb_config.name}_RUN_{self.run_id}")
            wandb.login(token = self.wandb_token) 
        if self.save_hf:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude):
                self.logger.info(f"Logging to hugging face repo {self.hf_repo_id}")
            hf_login(token = self.hf_token) 

        self.logger.setLevel(self.log_level)
        
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude):
            self.logger.info(f"Initializing {self.model_name.upper()}") 
        
        with open(os.path.join(self.save_checkpoint_path, f"RUN_{self.run_id}", \
                  f"RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_CONFIG.json"), 'w') as f:
            
            run_config_dict = {} 
            
            run_config_dict['train_config'] = asdict(self.train_config)
            run_config_dict['optimizer_config'] = asdict(self.optimizer_config)
            run_config_dict['scheduler_config'] = asdict(self.scheduler_config)
            run_config_dict['dataloader_config'] = asdict(self.dataloader_config)
            run_config_dict['model_config'] = asdict(self.model_config)
            run_config_dict['criterion_config'] = asdict(self.criterion_config)
            run_config_dict['wandb_config'] = asdict(self.wandb_config)
           
            json.dump(run_config_dict, f, indent = 4)
        
        self.train_data_root_path = self.dataloader_config.train_data_root_path
 
    def train(self):
        self._compile_warmup() 
        self._init_wandb() 
        self._check_device_warn()
        global_steps = 0 
        rank = dist.get_rank() 
        is_main_rank = rank == 0 
        X, y = get_data(self.train_data_root_path)
        
        self.dataloader = get_dataloader(X, y, parallel_type = self.parallel_type, 
                                         rank = dist.get_rank(), **self.train_dataloader_config)
        self._check_dataloader_sampler()      
       
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude):
            for epoch in range(self.epochs):
                progress_bar = tqdm(enumerate(self.dataloader), desc="Training", total=len(self.dataloader), 
                                    disable = (dist.get_rank()!=0 and self.parallel_type in ['fsdp', 'ddp']))
                for i, (X, y) in progress_bar:
                    X, y = X.to(self.device, non_blocking = True), y.to(self.device, non_blocking = True)
                if self.mixed_precision:
                    with autocast(device_type = 'cuda', dtype = self.mixed_precision_dtype()):
                        if is_main_rank:
                            start_time = time.perf_counter()
                        logits = self.model(X)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        pplx = torch.exp(loss)
                    loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx) 
                    self.scaler.scale(loss).backward() 
                    self.scaler.unscale_(self.optimizer)
                    if self.track_grad_norm:
                        grad_norm_dict = self._get_grad_norm()
                    self._clip_grad_norm()
                    self.scaler.step(self.optimizer) 
                    self.scaler.update()
                    self.scheduler.step()
                    if is_main_rank:
                        end_time = time.perf_counter()
                else:
                    if is_main_rank:
                        start_time = time.perf_counter()
                    logits = self.model(X) 
                    loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    pplx = torch.exp(loss)
                    loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx) 
                    loss.backward()
                    if is_main_rank:
                        end_time = time.perf_counter()
                    self._clip_grad_norm()
                    if self.track_grad_norm:
                        grad_norm_dict = self._get_grad_norm()
                    self.optimizer.step()
                    self.scheduler.step()
                
                global_steps += 1 
               
                if is_main_rank: 
                    progress_bar.set_description(desc = f"Epoch: {epoch + 1} | Local step: {i + 1} | \
                                                 Global step: {global_steps + 1} | LR: {self.scheduler.get_last_lr()[0]} | Loss: {loss_avg.item()} \
                                                 | pplx: {pplx_avg.item()} | Time: {end_time - start_time}") 
               
                if self.wandb_ and is_main_rank: 
                    wandb_dict = {
                        "loss": loss_avg.item(),
                        "perplexity": pplx_avg.item(),
                    }
                    
                    if self.track_grad_norm:
                        wandb_dict.update(grad_norm_dict)
                        
                    wandb.log(wandb_dict)  
                    
                if global_steps % self.checkpoint_steps == 0:
                    dist.barrier() 
                    self._clr_mem(gc_ = True, cuda_clr_cache = True, X = X, y = y, logits = logits)
                   
                    if is_main_rank: 
                        model_state_dict = self._get_model_state_dict()
                        optim_state_dict = self._get_optim_state_dict()  
                        scheduler_state_dict = self._get_scheduler_state_dict()
                      
                        self._save_checkpoint(
                            model_state_dict = model_state_dict,
                            optim_state_dict = optim_state_dict,
                            scheduler_state_dict = scheduler_state_dict,
                            epoch = epoch,
                            steps = i + 1,
                            global_steps = global_steps + 1,
                            save_hf = self.save_hf
                        ) 
                   
                if self.val_steps and (global_steps % self.val_steps == 0):
                    self.model.eval() 
                    if global_steps % self.checkpoint_steps != 0:
                        dist.barrier()
                        self._clr_mem(gc_ = True, cuda_clr_cache = True, X = X, y = y, logits = logits)
                    
                    val_dataloader = self._get_val_dataloader()
                    val_progress_bar = tqdm(enumerate(val_dataloader), desc = "Evaluating", total = len(val_dataloader),
                                        disable = (dist.get_rank()!=0 and self.parallel_type in ['fsdp', 'ddp']))
                   
                    val_steps = 0 
                    loss_accum = 0
                    pplx_accum = 0 
                  
                    with torch.no_grad(): 
                        for i, (X_val, y_val) in val_progress_bar:
                            X_val, y_val = X_val.to(self.device, non_blocking = True), y_val.to(self.device, non_blocking = True)
                            
                            if self.val_mixed_precision:
                                with autocast(device_type = 'cuda', dtype = self.val_mixed_precision_dtype):
                                    logits = self.model(X_val)
                                    loss = self.criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                                pplx = torch.exp(loss)
                                loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx)
                            else:
                                logits = self.model(X_val)
                                loss = self.criterion(logits.view(-1, logits.size(-1)), y_val.view(-1))
                                pplx = torch.exp(loss)
                                loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss, pplx)
                            
                            loss_accum += loss_avg
                            pplx_accum += pplx_avg
                            val_steps += 1

                    val_loss = loss_accum / val_steps
                    val_pplx = pplx_accum / val_steps

                    if self.wandb_:
                        wandb.log({
                            "val loss": val_loss.item(),
                            "val perplexity": val_pplx.item()
                            }
                        )
                   
                    self._clr_mem(
                        gc_ = True, 
                        cuda_clr_cache = True, 
                        X_val = X_val, 
                        y_val = y_val, 
                        logits = logits, 
                        val_dataloader = val_dataloader
                    ) 
                    
                    self.model.train()

        self._cleanup()
        self.cleanup()

    def cleanup(self):
        if hasattr(self, 'tee_handler') and self.tee_handler is not None:
            self.tee_handler.close()

    def _clip_grad_norm(self):
        if self.max_grad_norm: 
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
     
    def _get_grad_norm(self):
        if self.track_grad_norm:
            assert self.wandb_, ValueError('wandb_ must be set to True if you want to track the gradient norm') 
            grad_norm_dict = {}
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    clean_name = name.replace('_fsdp_wrapped_module.', '').replace('._flat_param', '')
                    param_norm = p.grad.norm(2)
                    grad_norm_dict[clean_name] = param_norm
            return grad_norm_dict 
            
    def _check_device_warn(self):
        if self.device.type == 'cpu':
            
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude):
                self.logger.warning('Training on CPU')
            
            cont = input('Continue [y/n]?')
            if cont.lower() == 'n':
                sys.exit(0)
                
    def _setup_parallel(self):
        dist.init_process_group(backend = 'nccl')
        local_rank = os.environ['LOCAL_RANK']
        try:
            torch.cuda.set_device(int(local_rank))
        except:
            raise Exception("Error setting device, LOCAL_RANK not found. \
                            Did you run `train.py` with torchrun?")
    
    def _cleanup(self):
        
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
            self.logger.info("Cleaning up Distributed Process Group") 
        dist.destroy_process_group()
        
    def _get_device(self):
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") if torch.cuda.is_available() else torch.device('cpu')
        return device
        
    def _check_dataloader_sampler(self):
        if self.parallel_type in ['fsdp', 'ddp']:
            if not isinstance(self.dataloader.sampler, DistributedSampler):
                raise ValueError('if parallel_type is fsdp or ddp, then the sampler of \
                                 the dataloader must DistributedSampler')
            
    def _get_model(self, model_config):
        with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
            self.logger.info("Initializing Model")
        model = LLaMA(**model_config)

        if self._compile:
            with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                self.logger.info("Compiling Model")
            model = torch.compile(model)

        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")

        if self.parallel_type == 'ddp':
            with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                self.logger.info("Wrapping model with DDP")
            model = model.cuda(local_rank)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
            with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                self.logger.info("Successfully wrapped model in DDP")
            return model

        elif self.parallel_type == 'fsdp':
            with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                self.logger.info("Wrapping model with FSDP")
            model = model.cuda(local_rank)

            if self.fsdp_wrap_policy == 'transformer':
                with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                    self.logger.info(f"Initializing FSDP with transformer policy")
                auto_wrap = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={TransformerBlock,},
                )
                model = FSDP(model, device_id=local_rank, auto_wrap_policy=auto_wrap)
            else:
                with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                    self.logger.info(f"Initializing FSDP with auto policy")
                model = FSDP(model, device_id=local_rank)

            with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                self.logger.info("Successfully wrapped model in FSDP")
            return model

        else:
            with supress_logging(logger=self.logger, disable=self.disable, disable_exclude=self.disable_exclude):
                self.logger.info("Not using parallelism")
            return model.cuda(local_rank)
            
    def _get_avg_rank_loss_pplx(self, loss, pplx):
        if self.parallel_type == 'ddp':
            loss_tensor = loss.detach().clone()
            dist.all_reduce(loss_tensor, op = ReduceOp.SUM)
            dist.all_reduce(pplx, op = ReduceOp.SUM)
            loss_avg = loss_tensor / dist.get_world_size() 
            pplx_avg = pplx / dist.get_world_size()
            return loss_avg, pplx_avg
        elif self.parallel_type == 'fsdp':
            dist.all_reduce(loss, op=ReduceOp.SUM)
            dist.all_reduce(pplx, op=ReduceOp.SUM)
            loss_avg = loss / dist.get_world_size()
            pplx_avg = pplx / dist.get_world_size()
            return loss_avg, pplx_avg
        
    def _get_model_state_dict(self):
        
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
            self.logger.info('Getting model state dict')    
        if self.parallel_type == 'ddp':
            return self.model.module.state_dict()
        elif self.parallel_type == 'fsdp':
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                return self.model.state_dict()
        else:
            if dist.get_rank() == 0:
                return self.model.state_dict()
        
    def _get_optim_state_dict(self):
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
            self.logger.info('Getting optimizer state dict')    
        if self.parallel_type == 'fsdp':
            return FSDP.optim_state_dict(self.model, self.optimizer, rank0_only=True)
        else:
            return self.optimizer.state_dict()

    def _get_scheduler_state_dict(self):
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
            self.logger.info('Getting scheduler state dict')
        if self.parallel_type in ['fsdp', 'ddp']:
            if dist.get_rank() == 0:
                return self.scheduler.state_dict()
            return {}  
        return self.scheduler.state_dict() 
    
    def _save_checkpoint(
        self,
        model_state_dict,
        optim_state_dict,
        scheduler_state_dict,
        epoch,
        steps,
        global_steps
        ):
       
        # save_checkpoint_root_path originiates from the train_config.json as save_checkpoint_root_path
        
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
            self.logger.info(f"Saving checkpoint at epoch {epoch} and global steps {global_steps}.")

        if dist.get_rank() == 0:
            root_path = os.path.join(self.save_checkpoint_path, f"RUN_{self.run_id}") # {save_checkpoint_root_path}/RUN_{self.run_id}
            os.makedirs(root_path, exist_ok = True)
            
            torch.save(
                {'epoch': epoch, 'global_steps': global_steps, 'model': model_state_dict,
                 'optim': optim_state_dict, 'scheduler_state_dict': scheduler_state_dict},
                f = os.path.join(root_path, f'RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_checkpoint_{epoch}_step_{steps}_global_steps_{global_steps}.pt')
            ) 
            
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                self.logger.info(f"Saved checkpoint at epoch {epoch} and global steps {global_steps}.")
           
            if self.save_hf and self.hf_repo_exists:
   
                with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                    self.logger.info(f"Saving checkpoint to hugging face at epoch {epoch} and global steps {global_steps}.")
    
                assert self.hf_repo_id is not None, ValueError('hf_repo_id must be specified')
                assert self.hf_root_path is not None, ValueError('hf_root_path must be specified')
               
                api = HfApi()
                
                try:
                    api.upload_file(
                        path_or_fileobj = os.path.join(root_path, \
                            f'RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_checkpoint_{epoch}_global_steps_{global_steps}.pt'),
                        path_in_repo = os.path.join(self.hf_root_path, f'RUN_{self.run_id}', \
                            f'RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_checkpoint_{epoch}_global_steps_{global_steps}.pt'),
                        repo_id = self.hf_repo_id,
                        repo_type = self.hf_repo_type if self.hf_repo_type else None,
                    )
                
                except Exception as e:
                   
                    with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                        self.logger.error(f"Failed to upload checkpoint to hugging face: {e}")
                        self.logger.error(f"Traceback: \n\n {traceback.format_exc()}")
                        raise
               
                try: 
                    api.upload_file(
                        path_or_fileobj = os.path.join(root_path, f"RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_CONFIG.json"),
                        path_in_repo = os.path.join(self.hf_root_path, f"RUN_{self.run_id}", \
                                                    f"RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_CONFIG.json"),
                        repo_id = self.hf_repo_id,
                        repo_type = self.hf_repo_type if self.hf_repo_type else None,
                        token = os.environ['HF_TOKEN']
                    )
                
                except Exception as e:
                    with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                        self.logger.error(f"Failed to upload config to hugging face: {e}")
                        self.logger.error(f"Traceback: \n\n {traceback.format_exc()}")
                        raise 
           
                with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                    self.logger.info(f"Saved checkpoint at epoch {epoch} and global steps {global_steps} to hugging face.")
           
            elif self.save_hf and not self.hf_repo_exists:
                
                with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                    self.logger.info(f"Creating hugging face repo at {self.hf_repo_id}") 
          
                assert self.hf_repo_id is not None, ValueError('hf_repo_id must be specified')
                assert self.hf_root_path is not None, ValueError('hf_root_path must be specified')
               
                try:
                    
                    with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                        self.logger.info(f"Creating hugging face repo at {self.hf_repo_id}") 
                    create_repo(
                        repo_id = self.hf_repo_id,
                        repo_type = self.hf_repo_type if self.hf_repo_type else None,
                        token = os.environ['HF_TOKEN']
                    )
                
                except Exception as e:
                    with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                        self.logger.error(f"Failed to create hugging face repo: {e}")
                        self.logger.error(f"Traceback: \n\n {traceback.format_exc()}")
                        raise
           
                with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                    self.logger.info(f"Created hugging face repo at {self.hf_repo_id}")
                
                self.hf_repo_exists = True 
                
                self._save_checkpoint(
                    model_state_dict = model_state_dict,
                    optim_state_dict = optim_state_dict,
                    scheduler_state_dict = scheduler_state_dict,
                    epoch = epoch,
                    steps = steps,
                    global_steps = global_steps
                )
          
    def _get_mixed_precision_dtype(self, mixed_precision_dtype=None):
      
        assert mixed_precision_dtype is not None, ValueError('mixed_precision_dtype must be specified')
        if mixed_precision_dtype.lower() == "bf16":
            return torch.bfloat16
        elif mixed_precision_dtype.lower() == "f16":
            return torch.float16
        else:
            raise ValueError(f"Invalid mixed precision dtype: {self.mixed_precision}, must be 'bf16' or 'f16'") 
            
    def _clr_mem(self, gc_ = False, cuda_clr_cache = True, *args, **kwargs):
        
        if gc_:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                self.logger.info('Collecting garbage.')
            gc.collect() 
        if cuda_clr_cache:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                self.logger.info('Clearing cuda cache.')
            torch.cuda.empty_cache()
        for i in args:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                self.logger.info(f'Deleting {i}')
            del i
        for key in kwargs:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                self.logger.info(f'Deleting {key}')
            del kwargs[key] 
           
    def _get_val_dataloader(self):
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
            self.logger.info('Loading validation data.') 
        X_val, y_val = get_data(self.val_data_root_path)
        
        val_dataloader = get_dataloader(
            X = X_val,
            y = y_val,
            batch_size = self.val_batch_size,
            num_workers = self.val_num_workers,
            shuffle = self.val_shuffle,
            pin_memory = self.val_pin_memory,
            parallelism_type = self.parallel_type,
            rank = dist.get_rank()
        ) 
        
        return val_dataloader
    
    def _init_wandb(self):
        if self.wandb_:
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                self.logger.info(f'Initializing wandb | Project: {self.wandb_config.project} | Run: {self.wandb_config.name}_RUN_{self.run_id}')
            assert isinstance(self.wandb_config, WandbConfig), ValueError('wandb_config must be type WandbConfig')
            self.wandb_config.name = self.wandb_config.name + "_RUN_" + self.run_id 

            wandb.init(
                **asdict(self.wandb_config)
            ) 
            with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
                self.logger.info('Initialized wandb')

    def _compile_warmup(self):
        if self._compile:
            with supress_logging(
                logger = self.logger, 
                disable = self.disable, 
                disable_exclude = self.disable_exclude): 
                self.logger.info('Running compile warmup')
            x = torch.randint(low = 0, high = self.vocab_size, size = (self.batch_size, self.context_length))
            for _ in tqdm(range(self._compile_warmup_steps), desc = 'Compile warmup.', total = self._compile_warmup_steps):
                self.model(x)
            self._clr_mem(gc_= True, cuda_clr_cache=True, x = x) 
            with supress_logging(
                logger = self.logger, 
                disable = self.disable, 
                disable_exclude = self.disable_exclude): 
                self.logger.info('Finished running compile warmup') 

    def _get_mixed_precision_dtype(self):
        if self.mixed_precision.lower() == "bf16":
            return torch.bfloat16
        elif self.mixed_precision.lower() == "f16":
            return torch.float16
        else:
            raise ValueError(f"Invalid mixed precision dtype: {self.mixed_precision}, must be 'bf16' or 'f16'")

    def _get_log_level(self):
        if self.log_level.lower() == 'info':
            return logging.INFO
        elif self.log_level.lower() == 'warning':
            return logging.WARNING
        elif self.log_level.lower() == 'error':
            return logging.ERROR
        elif self.log_level.lower() == 'critical':
            return logging.CRITICAL
        elif self.log_level.lower() == 'debug':
            return logging.DEBUG
        else:
            raise ValueError(f"Invalid log level: {self.log_level}")

    def setup_logger(self, log_level="INFO", log_root_path=None, return_date_time=False):
        import sys
        from io import StringIO

        class TeeHandler:
            def __init__(self, filename):
                self.file = open(filename, 'a')
                self.stdout = sys.stdout
                sys.stdout = self
                sys.stderr = self 

            def write(self, data):
                self.file.write(data)
                self.file.flush()
                self.stdout.write(data)
                
            def flush(self):
                self.file.flush()
                self.stdout.flush()
                
            def close(self):
                sys.stdout = self.stdout
                sys.stderr = sys.__stderr__ 
                self.file.close()

        self.logger.handlers = []
        self.logger.setLevel(log_level)

        console_handler = RichHandler(show_time=True, show_level=True, show_path=False)
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tee_handler = None

        if log_root_path and dist.is_initialized():
            os.makedirs(log_root_path, exist_ok=True)
            log_file = os.path.join(log_root_path, f"run_{date_time}.log")
        
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.tee_handler = TeeHandler(log_file)

        if return_date_time:
            return date_time

def get_scheduler(optimizer, warmup_steps, constant_steps, decay_steps, max_lr, min_lr, *args, **kwargs):
   
    if not min_lr <= optimizer.param_groups[0]['initial_lr'] <= max_lr:
        
        with supress_logging(logger = self.logger, disable = self.disable, disable_exclude = self.disable_exclude): 
            self.logger.warning(f"min_lr {min_lr} is not between optimizer's initial_lr "
                     f"{optimizer.param_groups[0]['initial_lr']} and max_lr {max_lr}")
    
    cycle_length = warmup_steps + constant_steps + decay_steps
    
    def lr_lambda(step):
        step_in_cycle = step % cycle_length
        
        if step_in_cycle < warmup_steps:
            return min_lr + (max_lr - min_lr) * (step_in_cycle / warmup_steps)
        
        elif step_in_cycle < warmup_steps + constant_steps:
            return max_lr
        
        else:
            decay_step = step_in_cycle - (warmup_steps + constant_steps)
            progress = decay_step / decay_steps
            decay = 0.5 * (1 + math.cos(math.pi * progress))
            return min_lr + (max_lr - min_lr) * decay
            
    return LambdaLR(optimizer, lr_lambda)

@contextmanager
def supress_logging(logger, disable=None, disable_exclude=None):
    
    if disable is None and disable_exclude is None:
        raise ValueError('Either disable or disable_exclude must be provided')

    if disable is not None and disable_exclude is not None:
        raise ValueError('Only one of disable or disable_exclude can be provided')

    rank = dist.get_rank() if dist.is_initialized() else 0
    suppress = False

    if disable is True:
        suppress = True
    elif isinstance(disable, int):
        suppress = (rank == disable)
    elif isinstance(disable_exclude, int):
        suppress = (rank != disable_exclude)

    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]

    if suppress:
        for h in stream_handlers:
            logger.removeHandler(h)
        try:
            yield
        finally:
            for h in stream_handlers:
                logger.addHandler(h)
    else:
        yield