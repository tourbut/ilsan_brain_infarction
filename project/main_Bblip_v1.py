import datetime
import hashlib
from model.Bblip_v1 import Brain_BLIP_pl
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import  WandbLogger
from dataset.dataset import Text_Image_DataModule
from model.eva_vit import Brain_BLIP_image
from pytorch_lightning.plugins.environments import ClusterEnvironment
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
)
from utils.utils import lambda_fn
import functools
import os 
import wandb

def __main__(): 
    ### make experiment ID 
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]

    #config = OmegaConf.load("./project/config/Brain_blip_v1_train_single_gpu_sample.yaml") 
    #config = OmegaConf.load("./project/config/Brain_blip_v1_train_single_gpu.yaml") 
    config = OmegaConf.load("./project/config/Brain_blip_v1_train_single_gpu_1000.yaml") 

    ### setting logger 
    wandb.login(key=config.wandb.API_KEY)
    logger = pl.loggers.WandbLogger(project="ilsan", name=f'{hash_key}')

    ### setting pytorch DDP 
    if config.pl_trainer.strategy == 'DDP': 
        strategy = pl.strategies.DDPStrategy(process_group_backend="nccl", find_unused_parameters=True)
    elif config.pl_trainer.strategy == 'DeepSpeed_Zero3_offload':
        strategy = pl.strategies.DeepSpeedStrategy(stage=3,  offload_optimizer={"device": "cpu"}, offload_parameters={"device": "cpu"})
    elif config.pl_trainer.strategy == 'FSDP': 
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
        strategy = pl.strategies.FSDPStrategy(cpu_offload=True, use_orig_params=True, auto_wrap_policy=auto_wrap_policy)
    else: 
        strategy = None

    ### setting pytorch lightning 
    pl.seed_everything(config.training_parameters.seed)
    DataModule = Text_Image_DataModule(config=config, img_dir=config.dataset.img_dir, meta_dir=config.dataset.meta_dir,prompt=config.prompt)
    
    ### initialize model 
    """
    Because of the compatibility between huggingface, torch lightning, and Deepspeed, 
    language model is initialized outside of the pytorch_lightning.Module().
    """
    lm_tokenizer = AutoTokenizer.from_pretrained(config.model.language_encoder.lm_model,
                                                 #device_map='sequential',
                                                 trust_remote_code=True, 
                                                 pad_token='<|extra_0|>')
    lm_tokenizer.pad_token_id = lm_tokenizer.eod_id
    lm_model = AutoModelForCausalLM.from_pretrained(config.model.language_encoder.lm_model, 
                                                    #device_map="sequential", 
                                                    fp16=True, 
                                                    trust_remote_code=True
                                                    ).eval()
    # disabling flash attention
    if config.model.language_encoder.use_flash_attn is False: 
        for i in range(len(lm_model.transformer.h)):
            lm_model.transformer.h[i].attn.use_flash_attn=False
            
    torch.cuda.empty_cache()
    model = Brain_BLIP_pl(config=config, lm_tokenizer=lm_tokenizer, lm_model=lm_model)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="valid_loss_total",
        mode="min",
        dirpath=config.pl_trainer.ckpt_dir,
        filename="{epoch:02d}-{valid_loss_total:.2f}",
    )
    
    
    ### initialize trainer 
    trainer = pl.Trainer(
        max_epochs=config.pl_trainer.max_epochs,
        devices=config.pl_trainer.devices,
        accelerator=config.pl_trainer.accelerator,
        num_nodes=config.pl_trainer.num_nodes,
        strategy=strategy,
        precision=config.pl_trainer.precision,
        logger=logger,
        log_every_n_steps=config.pl_trainer.log_every_n_steps, 
        callbacks=[checkpoint_callback],
    )

    # training 
    trainer.fit(model, datamodule=DataModule)

    # save model
    #trainer.save_checkpoint(f"{config.model.checkpoint_path}/{hash_key}", weights_only=True)
    

    

if __name__ == '__main__': 
    __main__()
