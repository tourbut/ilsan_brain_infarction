import datetime
import hashlib
from model.Bblip_t5_curriculum import Brain_BLIP_pl
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl 
from pytorch_lightning.loggers import  WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.dataset_curriculum import Text_Image_DataModule
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
)
from utils.utils import lambda_fn
import functools
import os 
import wandb

import warnings
warnings.filterwarnings('ignore')

def __main__(): 
    ### make experiment ID 
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]

    config = OmegaConf.load("./project/config/Brain_blip_t5_train_single_gpu_curriculum_setting2.yaml") 

    ### setting logger 
    wandb.login(key=config.wandb.API_KEY)
    logger = pl.loggers.WandbLogger(project="ilsan", name=f'{hash_key}')
    
    ### setting pytorch DDP 
    if config.pl_trainer.strategy == 'DP':
        strategy="dp" 
    elif config.pl_trainer.strategy == 'DDP': 
        strategy = pl.strategies.DDPStrategy(process_group_backend="nccl",find_unused_parameters=False,)
    elif config.pl_trainer.strategy == 'DeepSpeed_Zero3_offload':
        strategy = pl.strategies.DeepSpeedStrategy(stage=3,  offload_optimizer=True, offload_parameters=True)
    elif config.pl_trainer.strategy == 'FSDP': 
        auto_wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn)
        #strategy = pl.strategies.FSDPStrategy(cpu_offload=True)
        strategy = pl.strategies.FSDPStrategy(cpu_offload=True, use_orig_params=True, auto_wrap_policy=auto_wrap_policy)
    else: 
        strategy = None

    ### setting pytorch lightning 
    pl.seed_everything(config.training_parameters.seed)
    DataModule = Text_Image_DataModule(config_dataset=config.dataset, batch_size=config.training_parameters.batch_size)
    

    #lm_model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-7B-Chat', device_map="sequential", fp16=True, trust_remote_code=True).eval().to('cpu')
    #lm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", device_map='sequential')
    #lm_model = OPTModel.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16, device_map='sequential').eval().to('cpu')
    torch.cuda.empty_cache()
    
    if config.pl_trainer.resume_training:
        checkpoint_path=config.pl_trainer.ckpt_dir+'/'+config.pl_trainer.ckpt_file
        print(checkpoint_path)
        model = Brain_BLIP_pl.load_from_checkpoint(checkpoint_path=checkpoint_path, img_size=config.dataset.img_size)
    else:
        model = Brain_BLIP_pl(config=config, img_size=config.dataset.img_size) 
    #model.cuda()
        
    ### checkpoint 
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/total_loss",
        mode="min",
        #every_n_epochs=1,
        dirpath=config.pl_trainer.ckpt_dir,
        filename=hash_key+"_{epoch:02d}-val_loss{val/total_loss:.2f}-all_data-setting1",
    )

    ### initialize trainer 
    trainer = pl.Trainer(
        max_epochs=config.pl_trainer.max_epochs,
        devices=[config.pl_trainer.devices],
        accelerator=config.pl_trainer.accelerator,
        num_nodes=config.pl_trainer.num_nodes,
        log_every_n_steps=config.pl_trainer.log_every_n_steps,
        strategy=strategy,
        precision=config.pl_trainer.precision,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # training 
    trainer.fit(model, datamodule=DataModule)
    
    # test 
    trainer.test(ckpt_path="best",datamodule=DataModule)

if __name__ == '__main__': 
    __main__()
