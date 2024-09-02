import datetime
import hashlib
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl 
from transformers import AutoModelForCausalLM, AutoTokenizer


from model.Bblip_v1 import Brain_BLIP_pl
from dataset.dataset import Text_Image_DataModule


def __main__(): 

    config = OmegaConf.load("./project/config/Brain_blip_v1_train_single_gpu_sample.yaml") 
    hash_key= 'c52ad3'
    checkpoint_path=f"{config.model.checkpoint_path}/{hash_key}.ckpt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DataModule = Text_Image_DataModule(config=config, img_dir=config.dataset.img_dir, meta_dir=config.dataset.meta_dir,prompt=config.prompt)
    
    test_loader = DataModule.test_dataloader()
    
    lm_tokenizer = AutoTokenizer.from_pretrained(config.model.language_encoder.lm_model, device_map='sequential',trust_remote_code=True, pad_token='<|extra_0|>')
    lm_tokenizer.pad_token_id = lm_tokenizer.eod_id
    lm_model = AutoModelForCausalLM.from_pretrained(config.model.language_encoder.lm_model, device_map="sequential", fp16=True, trust_remote_code=True).eval().to('cpu')
        
    model = Brain_BLIP_pl(config=config, 
                          lm_tokenizer=lm_tokenizer, 
                          lm_model=lm_model).to(device)

    model.eval()
    
    
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            inputs = data
            #inputs = data.pop('label')
            labels = data['label']
            #inputs = inputs.to(device)
            #labels = labels.to(device)
            outputs = model(inputs)
            print('outputs',outputs)
            print('data',outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


if __name__ == '__main__': 
    __main__()
