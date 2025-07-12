import torch
import torch.nn as nn
import torch.optim as opt
import torch.distributed as dist
import torch.nn.functional as F
import logging
import itertools
import json
import math
import time
import os
import gc
import sys
import functools
import traceback
import wandb
import signal

from torchinfo import summary
from io import StringIO
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
from pyfiglet import Figlet
from termcolor import colored
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
        train_config:dict,
        debug_nccl:bool = False,
        debug_level:str = 'INFO',
        debug_subsys:bool = False,
        run_id:int = None,
        rm_logs_for_run_id:bool = False,
        ):
       
        self.model_config = ModelConfig(**model_config)
        self.criterion_config = CriterionConfig(**criterion_config)
        self.optimizer_config = OptimizerConfig(**optimizer_config)
        self.scheduler_config = SchedulerConfig(**scheduler_config)
        self.dataloader_config = DataloaderConfig(**dataloader_config)
        self.wandb_config = WandbConfig(**wandb_config)
        self.train_config = TrainingConfig(**train_config)

        self.log_level = self.train_config.log_level
        self.log_root_path = self.train_config.log_root_path if hasattr(self.train_config, 'log_root_path') else None

        self.debug_nccl = debug_nccl
        self.debug_level = debug_level
        self.debug_subsys = debug_subsys
 
        self._original_stderr = sys.__stderr__
        
        self.run_id = self.get_run_id()
       
        self.logger = logging.getLogger(__name__)
        self.rm_logs_for_run_id = rm_logs_for_run_id
        self.run_start_date_time = self.setup_logger(log_level=self.log_level, \
                                                     log_root_path=self.log_root_path, return_date_time = True)

        assert torch.cuda.is_available(), ValueError("No CUDA device found")
     
        self.train_dataloader_config = self.dataloader_config.train_dataloader_config 
        self.val_dataloader_config = self.dataloader_config.val_dataloader_config 
      
        self._save_on_interrupt = self.train_config.save_on_interrupt 
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
         
        track_grad_norm_state = self.train_config.track_grad_norm 
        self.track_grad_norm = self.train_config.track_grad_norm if self.train_config.wandb else False
        
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
        self.load_checkpoint = self.train_config.load_checkpoint
        self.load_checkpoint_path = self.train_config.load_checkpoint_path
        self.fsdp_wrap_policy = self.train_config.fsdp_wrap_policy
        self.model_name = self.model_config.model_name
        self.model_series_name = self.model_config.model_series_name

        if track_grad_norm_state != self.track_grad_norm:
            self.logger.warning("Gradient norm tracking is disabled because wandb is not enabled")
        
        load_dotenv(dotenv_path = 'main/configs/.env') 
        
        self.hf_token = os.environ.get('HF_TOKEN')
        self.wandb_token = os.environ.get('WANDB_API_KEY')
      
        if self.save_hf and self.hf_token is None:
            raise ValueError(f'HF_TOKEN must be specified if save_hf is True to save model checkpoints to Hugging Face repo {self.hf_repo_id}')

        if self.wandb_ and self.wandb_token is None:
            raise ValueError(f'WANDB_TOKEN must be specified if wandb is True, to log the training run to wandb project {self.wandb_config.project}')
            
        fig = Figlet(font='larry3d')
        
        self.scaler = GradScaler() if self.mixed_precision else None
        
        self._setup_nccl_logging() 
        self._setup_parallel()
       
        if not dist.is_initialized() or dist.get_rank() == 0:
            self._ascii_art_model_name()
       
        self.device = self._get_device()
        self.model = self._get_model(asdict(self.model_config))

        criterion_config_dict = asdict(self.criterion_config)
        optimizer_config_dict = asdict(self.optimizer_config)
        scheduler_config_dict = asdict(self.scheduler_config)

        del criterion_config_dict['extra_args']
        del optimizer_config_dict['extra_args']
        del scheduler_config_dict['extra_args']

        self.criterion = nn.CrossEntropyLoss(**criterion_config_dict)
        self.optimizer = opt.AdamW(self._build_param_groups(self.model, optimizer_config_dict),
                                   **{k: v for k, v in optimizer_config_dict.items() if k != 'weight_decay'})
        self.scheduler = self.get_scheduler(self.optimizer, **scheduler_config_dict)
    
        if self.load_checkpoint:
            assert self.load_checkpoint_path is not None, ValueError("load_checkpoint_path \
                                        must be specified if load_checkpoint is True")
            assert isinstance(self.load_checkpoint_path, str), ValueError("load_checkpoint_path \
                                        must be a string") 
            
            self._chk_cont_epoch, self._chk_cont_global_step, self._chk_cont_local_steps = \
                self._load_checkpoint(self.load_checkpoint_path) 
 
        if self.load_checkpoint_path is not None and not self.load_checkpoint:
            self.logger.warning("load_checkpoint_path is specified but load_checkpoint is False, will not start from checkpoint")
              
        
        if self.wandb_:
            self.logger.info(f"[Rank {self._get_local_rank()}] Logging to wandb project {self.wandb_config.project} as {self.wandb_config.name}_RUN_{self.run_id}")
            if not dist.is_initialized() or dist.get_rank() == 0:
                wandb.login(key = self.wandb_token) 
        if self.save_hf:
            self.logger.info(f"[Rank {self._get_local_rank()}] Logging to hugging face repo {self.hf_repo_id}")
            hf_login(token = self.hf_token) 

        self.logger.setLevel(self.log_level)
        
        self.logger.info(f"[Rank {self._get_local_rank()}] Initializing {self.model_name.upper()}") 
       
        os.makedirs(os.path.join(self.save_checkpoint_path, f"RUN_{self.run_id}"), exist_ok = True) 
           
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
        
        self.train_data_root_path = self.train_dataloader_config['train_data_root_path']

    def train(self):
        rank = dist.get_rank()
        is_main_rank = (rank == 0)
       
        z_save = False # for saving checkpoint, under signal_handler - will only save if z = True
        def signal_handler(signum, frame):
            self.logger.info("\nReceived interrupt signal. Cleaning up...")
            
            model_sd = self._get_model_state_dict()
            optimizer_sd = self._get_optimizer_state_dict()
            scheduler_sd = self._get_scheduler_state_dict() 
           
            if z_save and self._save_on_interrupt: 
                self._save_checkpoint(
                    model_sd = model_sd,
                    optimizer_sd = optimizer_sd,
                    scheduler_sd = scheduler_sd,
                    epoch = epoch,
                    global_steps = global_steps,
                    local_steps = local_steps
                ) 
        
            if is_main_rank:
                wandb.finish()
            
            self.cleanup()
            self._cleanup()
            os._exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        should_abort = torch.tensor(0, device=self.device)

        try:
            resume_step  = getattr(self, '_chk_cont_local_steps', 0)
            start_epoch  = getattr(self, '_chk_cont_epoch', 0)
            global_steps = getattr(self, '_chk_cont_global_step', 0)
            
            self._compile_warmup()

            if is_main_rank:
                self._init_wandb()
            
            self._check_device_warn()
            
            if global_steps:
                self.logger.info(f"[Rank {self._get_local_rank()}] Resuming training from epoch {start_epoch + 1}, global step {global_steps}")

            X, y = get_data(self.train_data_root_path)
            self.dataloader = get_dataloader(
                X, y,
                parallelism_type=self.parallel_type,
                rank=rank,
                batch_size=self.train_dataloader_config['train_batch_size'],
                num_workers=self.train_dataloader_config['train_num_workers'],
                shuffle=self.train_dataloader_config['train_shuffle'],
                pin_memory=self.train_dataloader_config['train_pin_memory'],
            )
            self._check_dataloader_sampler()

            world_size = int(dist.get_world_size())
            num_batches = len(self.dataloader)
            self.logger.info(f"[GLOBAL] Training {num_batches * world_size} batches per epoch over all RANKS.")
            self.logger.info(f"[GLOBAL] Training {num_batches * self.context_length * world_size} tokens per epoch over all RANKS.")
            self.logger.info(f"[GLOBAL] Training {num_batches * self.context_length * world_size * (self.epochs - start_epoch)} tokens over remaining epochs.")
            self.logger.info(f"[Rank {self._get_local_rank()}] Training {num_batches} batches per epoch.")
            self.logger.info(f"[Rank {self._get_local_rank()}] Training {num_batches * self.context_length} tokens per epoch.")
            self.logger.info(f"[Rank {self._get_local_rank()}] Training {num_batches * self.context_length * (self.epochs - start_epoch)} tokens over remaining epochs.")

            if self.debug_nccl or self.debug_subsys:
                self._setup_nccl_logging(_unset=True)

            os.system('cls' if os.name == 'nt' else 'clear')
            self._ascii_art_model_name()


            for epoch in range(start_epoch, self.epochs):
                z_save = True 
                self.logger.info(f"[Rank {self._get_local_rank()}] Starting epoch {epoch + 1}/{self.epochs}.")

                if hasattr(self.dataloader, 'sampler') and hasattr(self.dataloader.sampler, 'set_epoch'):
                    self.dataloader.sampler.set_epoch(epoch)

                if epoch == start_epoch and resume_step > 0:
                    data_iter     = itertools.islice(self.dataloader, resume_step, None)
                    total_batches = len(self.dataloader) - resume_step
                    self.logger.info(f"[Rank {self._get_local_rank()}] Resuming from batch {resume_step}")
                else:
                    data_iter = iter(self.dataloader)
                    total_batches = len(self.dataloader)

                progress_bar = tqdm(
                    data_iter,
                    total=total_batches,
                    desc=f"Epoch {epoch + 1}/{self.epochs} | Loss: - | PPLX: - | LR: -",
                    disable=(not is_main_rank and self.parallel_type in ['fsdp','ddp']),
                    ascii=False
                )
                local_steps = 0

                for X_batch, y_batch in progress_bar:
                    X_batch, y_batch = X_batch.to(self.device, non_blocking=True), y_batch.to(self.device, non_blocking=True)
                    local_steps  += 1
                    global_steps += 1

                    if self.mixed_precision:
                        with autocast(device_type='cuda', dtype=self.mixed_precision_dtype):
                            logits = self.model(X_batch)
                            loss   = self.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                        loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss)
                        self.scaler.scale(loss_avg).backward()
                        self.scaler.unscale_(self.optimizer)
                        if self.track_grad_norm:
                            grad_norm_dict = self._get_grad_norm()
                        self._clip_grad_norm()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                    else:
                        logits = self.model(X_batch)
                        loss   = self.criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
                        loss_avg, pplx_avg = self._get_avg_rank_loss_pplx(loss)
                        loss_avg.backward()
                        if self.track_grad_norm:
                            grad_norm_dict = self._get_grad_norm()
                        self._clip_grad_norm()
                        self.optimizer.step()
                        self.scheduler.step()

                    if is_main_rank:
                        lr = self.scheduler.get_last_lr()[0]
                        progress_bar.set_description(
                            f"Epoch {epoch + 1}/{self.epochs} | "
                            f"Loss: {loss_avg:.4f} | PPLX: {pplx_avg:.2f} | LR: {lr:.2e}"
                        )

                    if self.wandb_ and is_main_rank:
                        log_dict = {"loss": loss_avg, "perplexity": pplx_avg}
                        if self.track_grad_norm:
                            log_dict.update(grad_norm_dict)
                        self.logger.info(f'GLOBAL STEPS: {global_steps}')
                        wandb.log(log_dict, step=global_steps)

                    if global_steps % self.checkpoint_steps == 0:
                        dist.barrier()
                        self._clr_mem(gc_=True, cuda_clr_cache=True, X=X_batch, y=y_batch, logits=logits)
                        dist.barrier()

                        model_sd = self._get_model_state_dict()
                        optim_sd = self._get_optim_state_dict()
                        scheduler_sd = self._get_scheduler_state_dict()

                        if is_main_rank:
                            ckpt_dir = os.path.join(self.save_checkpoint_path, f"RUN_{self.run_id}")
                            ckpt_name= f"RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_E{epoch}_S{local_steps}_G{global_steps}.pt"
                            ckpt_path= os.path.join(ckpt_dir, ckpt_name)
                            progress_bar.set_description("Saving checkpoint...")
                            self._save_checkpoint(
                                model_state_dict = model_sd,
                                optim_state_dict = optim_sd,
                                scheduler_state_dict = scheduler_sd,
                                epoch = epoch,
                                steps = local_steps,
                                global_steps = global_steps,
                            )
                            progress_bar.set_description("Checkpoint saved")
                            
                        dist.barrier()

                    if self.val_steps and (global_steps % self.val_steps == 0):
                        self.model.eval() 
                        if global_steps % self.checkpoint_steps != 0:
                            dist.barrier()
                            self._clr_mem(gc_=True, cuda_clr_cache=True, X=X_batch, y=y_batch, logits=logits)
                        
                        val_dataloader = self._get_val_dataloader()
                        
                        val_console_handlers = [h for h in self.logger.handlers if isinstance(h, logging.StreamHandler)]
                        for _h in val_console_handlers:
                            self.logger.removeHandler(_h)
                        
                        val_progress_bar = tqdm(
                            enumerate(val_dataloader),
                            desc="Evaluating",
                            total=len(val_dataloader),
                            disable=(dist.get_rank() != 0 and self.parallel_type in ['fsdp', 'ddp']),
                            ascii=False
                        )
                        
                        total_val_loss = 0.0
                        total_val_examples = 0
                        
                        with torch.no_grad(): 
                            for i, (X_val, y_val) in val_progress_bar:
                                X_val, y_val = X_val.to(self.device, non_blocking=True), y_val.to(self.device, non_blocking=True)
                                if self.val_mixed_precision:
                                    with autocast(device_type='cuda', dtype=self.val_mixed_precision_or_dtype):
                                        if is_main_rank:
                                            start_time = time.perf_counter() 
                                        logits = self.model(X_val)
                                        if is_main_rank:
                                            end_time = time.perf_counter()
                                        loss = F.cross_entropy(
                                            input=logits.view(-1, logits.size(-1)), 
                                            target=y_val.view(-1),
                                            ignore_index=self.criterion_config.ignore_index,
                                            reduction='mean' 
                                        )
                                else:
                                    if is_main_rank:
                                        start_time = time.perf_counter() 
                                    logits = self.model(X_val)
                                    if is_main_rank:
                                        end_time = time.perf_counter()
                                    loss = F.cross_entropy(
                                        input=logits.view(-1, logits.size(-1)), 
                                        target=y_val.view(-1),
                                        ignore_index=self.criterion_config.ignore_index,
                                        reduction='mean'     
                                    )
                                batch_size = X_val.size(0)
                                total_val_loss += loss * batch_size
                                total_val_examples += batch_size
                                if is_main_rank:
                                    val_progress_bar.set_description(f"Evaluating | Time: {end_time - start_time:.5f}s")

                        total_val_loss = total_val_loss.clone().detach()
                        dist.all_reduce(total_val_loss, op=dist.ReduceOp.SUM)
                        
                        total_val_examples = torch.tensor(total_val_examples, dtype=torch.float, device=self.device)
                        dist.all_reduce(total_val_examples, op=dist.ReduceOp.SUM)
                        
                        global_avg_val_loss = total_val_loss / total_val_examples
                        global_avg_val_pplx = torch.exp(global_avg_val_loss)
                        
                        if self.wandb_ and is_main_rank:
                            wandb.log({
                                "val_loss": global_avg_val_loss.item(),
                                "val_perplexity": global_avg_val_pplx.item()
                            }, step=global_steps)  
                    
                        self._clr_mem(
                            gc_=True, 
                            cuda_clr_cache=True, 
                            X_val=X_val, 
                            y_val=y_val
                        ) 
                        
                        for _h in val_console_handlers:
                            self.logger.addHandler(_h)
                        
                        self.model.train()
                
                dataloader_iter = enumerate(self.dataloader)

                self._restore_console_logging()

                if self.load_checkpoint:
                    resume_step = 0
            
            model_sd = self._get_model_state_dict()
            optim_sd = self._get_optim_state_dict()
            scheduler_sd = self._get_scheduler_state_dict() 
            
            self._save_checkpoint(
                model_state_dict = model_sd,
                optim_state_dict = optim_sd,
                scheduler_state_dict = scheduler_sd,
                epoch = self.epochs,
                global_steps = global_steps,
                local_steps = local_steps,
            ) 

            self._cleanup()
            self.cleanup()
        
        except Exception as e:
            self.logger.error(f"[Rank {self._get_local_rank()}] Training failed with exception: {str(e)}")
            self._cleanup()
            self.cleanup()
            raise

    def cleanup(self):
        if hasattr(self, 'tee_handler') and self.tee_handler is not None:
            self.tee_handler.close()
        self._setup_nccl_logging(_unset=True)

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
            
            self.logger.warning(f'[Rank {self._get_local_rank()}] Training on CPU')
            
            cont = input('Continue [y/n]?')
            if cont.lower() == 'n':
                sys.exit(0)
                
    def _setup_parallel(self):
        if not dist.is_initialized():
            try:
                local_rank = int(os.environ['LOCAL_RANK'])
                rank = int(os.environ['RANK'])
                world_size = int(os.environ['WORLD_SIZE'])
            except KeyError as e:
                raise RuntimeError(
                    f"Missing required environment variable: {str(e)}. "
                    "Make sure you're launching with torchrun."
                )

            torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')

            try:
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    rank=rank,
                    world_size=world_size,
                    device_id = torch.device(f'cuda:{local_rank}')
                )
                self.logger.info(f"[Rank {rank}] Successfully initialized training process group on GPU {local_rank}")
            except Exception as e:
                self.logger.error(f"[Rank {rank}] Error initializing training process group: {str(e)}")
                raise RuntimeError(
                    f"Error initializing distributed training: {str(e)}\n"
                    "Make sure you're running with torchrun and have set all required environment variables."
                )
    
    def _cleanup(self):
        
        self.logger.info(f"[Rank {self._get_local_rank()}] Cleaning up Distributed Process Group") 
        dist.destroy_process_group()
        
    def _get_device(self):
        if dist.is_initialized():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            return torch.device(f"cuda:{local_rank}")
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def _check_dataloader_sampler(self):
        if self.parallel_type in ['fsdp', 'ddp']:
            if not isinstance(self.dataloader.sampler, DistributedSampler):
                raise ValueError('if parallel_type is fsdp or ddp, then the sampler of \
                                 the dataloader must DistributedSampler')
            
    def _get_model(self, model_config):
        self.logger.info(f"[Rank {self._get_local_rank()}] Initializing Model")
        model = LLaMA(**model_config)

        if self._compile:
            self.logger.info(f"[Rank {self._get_local_rank()}] Compiling Model")
            model = torch.compile(model)

        if self.parallel_type == 'ddp':
            
            self.logger.info(f"[Rank {self._get_local_rank()}] Wrapping model with DDP at rank {int(os.environ.get('LOCAL_RANK', 0))}") 
            model = model.cuda(int(os.environ.get('LOCAL_RANK', 0)))
            model = DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK', 0))], output_device=int(os.environ.get('LOCAL_RANK', 0)))
            self.logger.info(f"[Rank {self._get_local_rank()}] Successfully wrapped model in DDP at rank {int(os.environ.get('LOCAL_RANK', 0))}")
            return model

        elif self.parallel_type == 'fsdp':
            self.logger.info(f"[Rank {self._get_local_rank()}] Wrapping model with FSDP at rank {int(os.environ.get('LOCAL_RANK', 0))}")
            model = model.cuda(int(os.environ.get('LOCAL_RANK', 0)))

            if self.fsdp_wrap_policy == 'transformer':
                self.logger.info(f"[Rank {self._get_local_rank()}] Initializing FSDP with transformer policy at rank {int(os.environ.get('LOCAL_RANK', 0))}")
                auto_wrap = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={TransformerBlock,},
                )
                model = FSDP(model, device_id=int(os.environ.get('LOCAL_RANK', 0)), auto_wrap_policy=auto_wrap)
            else:
                self.logger.info(f"[Rank {self._get_local_rank()}] Initializing FSDP with auto policy at rank {int(os.environ.get('LOCAL_RANK', 0))}")
                model = FSDP(model, device_id=int(os.environ.get('LOCAL_RANK', 0)))

            self.logger.info(f"[Rank {self._get_local_rank()}] Successfully wrapped model in FSDP at rank {int(os.environ.get('LOCAL_RANK', 0))}")
            return model

        else:
            self.logger.info(f"[Rank {self._get_local_rank()}] Not using parallelism")
            return model.cuda(int(os.environ.get('LOCAL_RANK', 0)))
            
    def _get_avg_rank_loss_pplx(self, loss, _val = None):
        if _val is not None:
            dist.all_reduce(loss, op = ReduceOp.SUM)  
            loss = loss / dist.get_world_size() 
            pplx = torch.exp(loss).item()
            return loss, pplx 
        elif _val is None:  
            loss_avg_scalar = loss.mean() 
            dist.all_reduce(loss_avg_scalar, op = ReduceOp.SUM) 
            loss_avg_scalar = loss_avg_scalar / dist.get_world_size() 
            pplx_avg = torch.exp(loss_avg_scalar).item()
            return loss_avg_scalar, pplx_avg
        
    def _get_model_state_dict(self):
       
        self.logger.info(f'[Rank {self._get_local_rank()}] Getting model state dict')    
        if self.parallel_type == 'ddp':
            state_dict = self.model.module.state_dict()
            dist.barrier()
            self.logger.info(f'[Rank {self._get_local_rank()}] Got model state dict')
            return state_dict
        elif self.parallel_type == 'fsdp':
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                state_dict = self.model.state_dict()
                dist.barrier()
                self.logger.info(f'[Rank {self._get_local_rank()}] Got model state dict')
                return state_dict
        else:
            if dist.get_rank() == 0:
                return self.model.state_dict()
        
    def _get_optim_state_dict(self):
        self.logger.info(f'[Rank {self._get_local_rank()}] Getting optimizer state dict')
        if self.parallel_type == 'fsdp':
            state_dict = FSDP.full_optim_state_dict(self.model, self.optimizer, rank0_only=True) 
            dist.barrier()
            self.logger.info(f'[Rank {self._get_local_rank()}] Got optimizer state dict')
            return state_dict
        else:
            state_dict = self.optimizer.state_dict()
            dist.barrier()
            self.logger.info(f'[Rank {self._get_local_rank()}] Got optimizer state dict')
            return state_dict

    def _get_scheduler_state_dict(self):
        self.logger.info(f'[Rank {self._get_local_rank()}] Getting scheduler state dict')
        if self.parallel_type in ['fsdp', 'ddp']:
            state_dict = self.scheduler.state_dict() 
            dist.barrier()
            self.logger.info(f'[Rank {self._get_local_rank()}] Got scheduler state dict')
            return state_dict
        return {}  
    
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

        root_path = os.path.join(self.save_checkpoint_path, f"RUN_{self.run_id}") # {save_checkpoint_root_path}/RUN_{self.run_id}
        os.makedirs(root_path, exist_ok=True)
      
        self.logger.info(f"[Rank {self._get_local_rank()}] Saving checkpoint at epoch {epoch} and global steps {global_steps}.") 
       
        torch.save(
                {'epoch': epoch, 'global_steps': global_steps, 'local_steps': steps, 'model_state_dict': model_state_dict,
                 'optim_state_dict': optim_state_dict, 'scheduler_state_dict': scheduler_state_dict},
                f = os.path.join(root_path, f'RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_EPOCH_{epoch}_STEP_{steps}_GLOBAL_STEPS_{global_steps}.pt')
            ) 
          
        del model_state_dict, optim_state_dict, scheduler_state_dict 
           
        self.logger.info(f"[Rank {self._get_local_rank()}] Saved checkpoint at epoch {epoch} and global steps {global_steps}.") 
        
        self.logger.info(f"[Rank {self._get_local_rank()}] Saved checkpoint at epoch {epoch} and global steps {global_steps}.")
       
        # main/checkpoints/RUN_001/RUN_001_DATETIME_2025-07-03_21-19-58_EPOCH_0_step_10_global_steps_10.pt 
        
        if self.save_hf and self.hf_repo_exists:

            self.logger.info(f"[Rank {self._get_local_rank()}] Saving checkpoint to hugging face at epoch {epoch} and global steps {global_steps}.")

            assert self.hf_repo_id is not None, ValueError('hf_repo_id must be specified')
            assert self.hf_root_path is not None, ValueError('hf_root_path must be specified')
            
            api = HfApi()
            
            try: # save to hugging face
                
                self.logger.info(f"[Rank {self._get_local_rank()}] Saving checkpoint to hugging face at epoch {epoch} and global steps {global_steps}.")
               
                api.upload_file(
                    path_or_fileobj = os.path.join(root_path, \
                        f'RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_EPOCH_{epoch}_STEP_{steps}_GLOBAL_STEPS_{global_steps}.pt'),
                    path_in_repo = os.path.join(self.hf_root_path, f'RUN_{self.run_id}', \
                        f'RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_EPOCH_{epoch}_STEP_{steps}_GLOBAL_STEPS_{global_steps}.pt'),
                    repo_id = self.hf_repo_id,
                    repo_type = self.hf_repo_type if self.hf_repo_type else None,
                )
                
                self.logger.info(f"[Rank {self._get_local_rank()}] Saved checkpoint to hugging face at epoch {epoch} and global steps {global_steps}.")
            
            except Exception as e:
                
                self.logger.error(f"[Rank {self._get_local_rank()}] Failed to upload checkpoint to hugging face: {e}")
                self.logger.error(f"[Rank {self._get_local_rank()}] Traceback: \n\n {traceback.format_exc()}")
                raise
            
            try: 
                self.logger.info(f"[Rank {self._get_local_rank()}] Uploading config to hugging face at epoch {epoch} and global steps {global_steps}.")
                api.upload_file(
                    path_or_fileobj = os.path.join(root_path, f"RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_CONFIG.json"),
                    path_in_repo = os.path.join(self.hf_root_path, f"RUN_{self.run_id}", \
                                                f"RUN_{self.run_id}_DATETIME_{self.run_start_date_time}_CONFIG.json"),
                    repo_id = self.hf_repo_id,
                    repo_type = self.hf_repo_type if self.hf_repo_type else None,
                    token = os.environ['HF_TOKEN']
                )
                self.logger.info(f"[Rank {self._get_local_rank()}] Uploaded config to hugging face at epoch {epoch} and global steps {global_steps}.")
            
            except Exception as e:
                self.logger.error(f"[Rank {self._get_local_rank()}] Failed to upload config to hugging face: {e}")
                self.logger.error(f"[Rank {self._get_local_rank()}] Traceback: \n\n {traceback.format_exc()}")
                raise 
        
            self.logger.info(f"[Rank {self._get_local_rank()}] Saved checkpoint at epoch {epoch} and global steps {global_steps} to hugging face.")
       
        elif self.save_hf and not self.hf_repo_exists:
            
            self.logger.info(f"[Rank {self._get_local_rank()}] Creating hugging face repo at {self.hf_repo_id}") 
        
            assert self.hf_repo_id is not None, ValueError('hf_repo_id must be specified')
            assert self.hf_root_path is not None, ValueError('hf_root_path must be specified')
            
            try:
                
                self.logger.info(f"[Rank {self._get_local_rank()}] Creating hugging face repo at {self.hf_repo_id}") 
                
                create_repo(
                    repo_id = self.hf_repo_id,
                    repo_type = self.hf_repo_type if self.hf_repo_type else None,
                    token = os.environ['HF_TOKEN']
                )
            
                self.logger.info(f"[Rank {self._get_local_rank()}] Created hugging face repo at {self.hf_repo_id}") 
            
            except Exception as e:
                self.logger.error(f"[Rank {self._get_local_rank()}] Failed to create hugging face repo: {e}")
                self.logger.error(f"[Rank {self._get_local_rank()}] Traceback: \n\n {traceback.format_exc()}")
                raise
        
            self.logger.info(f"[Rank {self._get_local_rank()}] Created hugging face repo at {self.hf_repo_id}")
            
            self.hf_repo_exists = True 
            
            self._save_checkpoint(
                model_state_dict = model_state_dict,
                optim_state_dict = optim_state_dict,
                scheduler_state_dict = scheduler_state_dict,
                epoch = epoch,
                steps = steps,
                global_steps = global_steps
            )
            
            del model_state_dict, optim_state_dict, scheduler_state_dict
          
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
            self.logger.info(f'[Rank {self._get_local_rank()}] Collecting garbage.')
            gc.collect() 
        if cuda_clr_cache:
            self.logger.info(f'[Rank {self._get_local_rank()}] Clearing cuda cache.')
            torch.cuda.empty_cache()
        for i in args:
            del i
        for key in list(kwargs.keys()):
            del kwargs[key] 
           
    def _get_val_dataloader(self):
        self.logger.info(f'[Rank {self._get_local_rank()}] Loading validation data.') 
        X_val, y_val = get_data(self.val_data_root_path)
        
        val_dataloader = get_dataloader(
            X = X_val,
            y = y_val,
            batch_size = self.val_batch_size,
            num_workers = self.val_num_workers,
            shuffle = self.val_shuffle,
            pin_memory = self.val_pin_memory,
            parallelism_type = self.parallel_type,
            rank = dist.get_rank(),
        ) 
        
        return val_dataloader
    
    def _init_wandb(self):
        if self.wandb_:
            if dist.is_initialized() and dist.get_rank() != 0:
                return
            
            self.logger.info(f'[Rank {self._get_local_rank()}] Initializing wandb | Project: {self.wandb_config.project} | Run: {self.wandb_config.name}_RUN_{self.run_id} at local_rank {self._get_local_rank()}')
            assert isinstance(self.wandb_config, WandbConfig), ValueError('wandb_config must be type WandbConfig')
            self.wandb_config.name = self.wandb_config.name + "_RUN_" + self.run_id 

            wandb_config_dict = asdict(self.wandb_config)
            del wandb_config_dict['extra_args']

            project = self.wandb_config.project 
            name = self.wandb_config.name 
            entity = self.wandb_config.entity
            tags = self.wandb_config.tags
            notes = self.wandb_config.notes
            id_ = self.wandb_config.id # OPTIONAL PARAMETER - if in thew andb_config.json, i don't have this specified we won't resume from the same fun
            
            wandb.init(
                project = project,
                name = name,
                entity = entity,
                tags = tags,
                notes = notes,
                id = id_,
                resume_from = f'{id_}?_step={self._chk_cont_global_step}' if self.load_checkpoint else None # resume from a checkpoing if we definfed it in the train_config.josn.
            ) 
            
            self.logger.info(f'[Rank {self._get_local_rank()}] Initialized wandb at local_rank {self._get_local_rank()}')

    def _compile_warmup(self):
        if self._compile:
            self.logger.info(f"[Rank {self._get_local_rank()}] Running compile warmup")
            x = torch.randint(low = 0, high = self.vocab_size, size = (self.batch_size, self.context_length))
            for _ in tqdm(range(self._compile_warmup_steps), desc = 'Compile warmup.', total = self._compile_warmup_steps):
                self.model(x)
            self._clr_mem(gc_= True, cuda_clr_cache=True, x = x) 
            self.logger.info(f"[Rank {self._get_local_rank()}] Finished running compile warmup")

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

        self.logger.handlers = []
        self.logger.setLevel(log_level)

        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.tee_handler = None

        if log_root_path and (not dist.is_initialized() or dist.get_rank() == 0):
            
            os.makedirs(os.path.join(log_root_path, f"RUN_{self.run_id}"), exist_ok=True)

            if self.rm_logs_for_run_id:
                self._rm_logs_for_run_id()
           
            log_file = os.path.join(log_root_path, f"RUN_{self.run_id}", f"run_{date_time}.log")
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.tee_handler = TeeHandler(log_file)
            self._log_file_path = log_file

        if return_date_time:
            return date_time

    def get_scheduler(self, optimizer, warmup_steps, constant_steps, decay_steps, max_lr, min_lr, *args, **kwargs):
        base_lr = optimizer.defaults["lr"]

        if not min_lr <= base_lr <= max_lr:
            self.logger.warning(
                f"[Rank {self._get_local_rank()}] base_lr {base_lr} is not between min_lr {min_lr} and max_lr {max_lr}"
            )

        cycle_length = warmup_steps + constant_steps + decay_steps

        def lr_lambda(step):
            step_in_cycle = min(step, cycle_length)

            if step_in_cycle < warmup_steps:
                lr = min_lr + (max_lr - min_lr) * (step_in_cycle / warmup_steps)

            elif step_in_cycle < warmup_steps + constant_steps:
                lr = max_lr

            else:
                decay_step = step_in_cycle - (warmup_steps + constant_steps)
                progress = decay_step / decay_steps
                decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = min_lr + (max_lr - min_lr) * decay

            return lr / base_lr

        return LambdaLR(optimizer, lr_lambda)

    def _get_local_rank(self):
        return int(os.environ['LOCAL_RANK'])

    def get_run_id(self):
        was_initialized = dist.is_initialized()
        original_backend = dist.get_backend() if was_initialized else None
        
        if not was_initialized and 'LOCAL_RANK' in os.environ:
            print(f'[Rank {self._get_local_rank()}] Initializing Temporary Process Group') 
            dist.init_process_group(backend='gloo', init_method='env://')
            should_destroy = True
        else:
            should_destroy = False
        
        try:
            if 'LOCAL_RANK' in os.environ:
                local_rank = int(os.environ['LOCAL_RANK'])
                if local_rank == 0:
                    user_input = input("Enter Run ID (3 digit integer, e.g. 001): ")
                    input_tensor = torch.tensor([ord(c) for c in user_input], dtype=torch.int64)
                    input_len = torch.tensor([len(user_input)], dtype=torch.int64)
                    dist.broadcast(input_len, src=0)
                    dist.broadcast(input_tensor, src=0)
                else:
                    input_len = torch.tensor([0], dtype=torch.int64)
                    dist.broadcast(input_len, src=0)
                    input_tensor = torch.zeros(input_len.item(), dtype=torch.int64)
                    dist.broadcast(input_tensor, src=0)
                    user_input = ''.join(chr(c) for c in input_tensor.tolist())
                    
                dist.barrier()
                return user_input
            else:
                return input("Enter Run ID (3 digit integer, e.g. 001): ")
                
        finally:
            if should_destroy:
                print(f'[Rank {self._get_local_rank()}] Destroying Temporary Process Group') 
                dist.destroy_process_group()
            if was_initialized and original_backend and original_backend != 'gloo':
                print(f'[Rank {self._get_local_rank()}] Restoring Original Process Group') 
                dist.init_process_group(backend=original_backend, init_method='env://')
            
    def _load_checkpoint(self, checkpoint_path):
        self.logger.info(f"[Rank {self._get_local_rank()}] Loading checkpoint from {checkpoint_path}") 
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.logger.info(f"[Rank {self._get_local_rank()}] Checkpoint loaded") 
        self.logger.info(f"[Rank {self._get_local_rank()}] Loading model state dict") 
        model_state_dict = checkpoint.get('model', checkpoint.get('model_state_dict'))
        self.logger.info(f"[Rank {self._get_local_rank()}] Model state dict loaded") 
        
        if model_state_dict is None:
            raise KeyError("No model state dict found in checkpoint")
        self.model.load_state_dict(model_state_dict)
       
        self.logger.info(f"[Rank {self._get_local_rank()}] Loading optimizer state dict") 
        optim_state_dict = checkpoint.get('optim', checkpoint.get('optim_state_dict'))
        if optim_state_dict is not None:
            self.logger.info(f"[Rank {self._get_local_rank()}] Optimizer state dict loaded")
            
            chk_groups = optim_state_dict.get('param_groups', [])
            cur_groups = self.optimizer.param_groups

            if len(chk_groups) != len(cur_groups):
                self.logger.warning(
                    "Skipping optimizer state load: checkpoint has %d param groups but current "
                    "optimizer expects %d. Starting with a fresh optimizer state.",
                    len(chk_groups), len(cur_groups)
                )
            else:
                try:
                    if self.parallel_type == 'fsdp' and optim_state_dict is not None:
                        try:
                            optim_state_dict = FSDP.optim_state_dict_to_load(self.model, self.optimizer, optim_state_dict)
                        except Exception as e:
                            self.logger.warning(f"[Rank {self._get_local_rank()}] Could not convert FSDP optimizer state dict: {e}")
                    self.optimizer.load_state_dict(optim_state_dict)
                except (ValueError, KeyError) as e:
                    self.logger.warning(
                        f"Could not load optimizer state: {str(e)}. Starting with fresh optimizer state."
                    )
       
        self.logger.info(f"[Rank {self._get_local_rank()}] Loading scheduler state dict") 
        scheduler_state_dict = checkpoint.get('scheduler_state_dict')
        if scheduler_state_dict is not None:
            self.logger.info(f"[Rank {self._get_local_rank()}] Scheduler state dict loaded")
            try:
                self.scheduler.load_state_dict(scheduler_state_dict)
            except (ValueError, KeyError) as e:
                self.logger.warning(f"Could not load scheduler state: {str(e)}. Starting with fresh scheduler state.")
       
        global_steps = checkpoint['global_steps'] 
        local_steps = checkpoint['local_steps']
        epoch = checkpoint['epoch']
       
        assert global_steps, KeyError("global_steps not found in checkpoint") 
        assert local_steps, KeyError("local_steps not found in checkpoint") 
        assert epoch, KeyError("epoch not found in checkpoint") 
        
        return (
            epoch,
            global_steps,
            local_steps
        )

    def _build_param_groups(self, model: nn.Module, optim_cfg: dict):
        """Create a deterministic AdamW param-group list.

        Groups:
          1. params with weight decay (all parameters except biases & 1-D like layer-norm weights)
          2. params without weight decay (biases & layer-norm/embedding weights)

        This matches the commonly used GPT/LLaMA weight-decay scheme and guarantees
        we always end up with exactly two param groups, independent of model size,
        so `load_state_dict` never complains about a group-count mismatch.
        """

        weight_decay = optim_cfg.get("weight_decay", 0.0)

        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen params
            if param.ndim >= 2 and not name.endswith("bias"):
                decay.append(param)
            else:
                no_decay.append(param)

        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def _restore_console_logging(self):
        if hasattr(self, 'console_handlers'):
            for handler in self.console_handlers:
                if handler not in self.logger.handlers:  
                    self.logger.addHandler(handler)

    def _ascii_art_model_name(self):
        fig = Figlet(font='larry3d')
        ascii_art = fig.renderText(f"{self.model_series_name}")
        colored_art = colored(ascii_art, color='red')
        print(colored_art, flush=True)

    def _setup_nccl_logging(self, _unset=False):
        rank = self._get_local_rank()
        self.logger.info(f"[Rank {rank}] Setting up NCCL logging")

        if _unset:
            self.logger.info(f"[Rank {rank}] Unsetting NCCL logging environment variables")
            for var in ['NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_DEBUG_FILE']:
                os.environ.pop(var, None)
            self.logger.info(f"[Rank {rank}] NCCL logging disabled")
            return

        if self.debug_nccl or self.debug_subsys:
            if self.debug_nccl:
                os.environ['NCCL_DEBUG'] = self.debug_level
            if self.debug_subsys:
                os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

            if self._log_file_path is not None:
                log_dir = os.path.dirname(self._log_file_path)
                self._nccl_log_file_path = os.path.join(
                    log_dir,
                    f"nccl_rank{rank}_pid{os.getpid()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
                )

                try:
                    with open(self._nccl_log_file_path, 'a'):
                        pass
                    os.environ['NCCL_DEBUG_FILE'] = self._nccl_log_file_path
                    self.logger.info(f"[Rank {rank}] Created NCCL log file at {self._nccl_log_file_path}")
                except OSError as err:
                    self.logger.warning(f"[Rank {rank}] Could not create NCCL log file: {err}")

            self.logger.info(f"[Rank {rank}] NCCL_DEBUG={os.environ.get('NCCL_DEBUG', 'not set')}")
            self.logger.info(f"[Rank {rank}] NCCL_DEBUG_SUBSYS={os.environ.get('NCCL_DEBUG_SUBSYS', 'not set')}")
            self.logger.info(f"[Rank {rank}] NCCL_DEBUG_FILE={os.environ.get('NCCL_DEBUG_FILE', 'not set')}")
            
    '''
    def _start_nccl_tail(self):
        """Spawn a daemon thread that tails the NCCL log file and mirrors its
        content to the original stderr so NCCL messages appear in the
        terminal while still being written to the file by NCCL itself."""
        
        if not hasattr(self, '_nccl_log_file_path') or not self._nccl_log_file_path:
            self.logger.warning("No NCCL log file path available for tailing")
            return
            
        if not os.path.exists(self._nccl_log_file_path):
            self.logger.warning(f"NCCL log file does not exist: {self._nccl_log_file_path}")
            return

        def _tail():
            try:
                self.logger.info(f"Starting NCCL log tailing from {self._nccl_log_file_path}")
                with open(self._nccl_log_file_path, 'r') as f:
                    f.seek(0, 2)
                    
                    while True:
                        current_position = f.tell()
                        line = f.readline()
                        
                        if not line:
                            time.sleep(0.1)
                            try:
                                if os.path.getsize(self._nccl_log_file_path) < current_position:
                                    f.seek(0)
                            except (IOError, OSError) as e:
                                self.logger.warning(f"Error checking log file size: {e}")
                                break
                            continue
                            
                        try:
                            self._original_stderr.write(f"[NCCL] {line}")
                            self._original_stderr.flush()
                        except (IOError, OSError) as e:
                            self.logger.error(f"Error writing to stderr: {e}")
                            break
                            
            except Exception as e:
                self.logger.error(f"NCCL tail thread error: {e}")
            finally:
                self.logger.info("NCCL log tailing thread terminated")

        t = threading.Thread(target=_tail, daemon=True, name="NCCL-Log-Tail")
        t.start()
    '''

    def _rm_logs_for_run_id(self):
        self.logger.info(f"Removing logs for run id: {self.run_id}")
        if self.rm_logs_for_run_id:
            logs_dir = os.listdir(self.log_root_path)  
            for pth in logs_dir:    
                full_pth = os.path.join(self.log_root_path, pth)
                if os.path.isfile(full_pth):
                    os.remove(full_pth)
        self.logger.info(f"Removed logs for run id: {self.run_id}")

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
        
    def isatty(self):
        return self.stdout.isatty()  
        
    def close(self):
        if hasattr(self, 'file'):
            sys.stdout = self.stdout
            sys.stderr = sys.__stderr__
            self.file.close()