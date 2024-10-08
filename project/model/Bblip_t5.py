# ref 1: https://github.com/Qybc/MedBLIP/blob/main/medblip/modeling_medblip_biomedlm.py
# ref 2: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
# ref 3: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py
# ref 4: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_t5_instruct.py
# ref 5: https://github.com/QwenLM/Qwen/blob/main/finetune.py#L172


import os 
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import torch.distributed as dist
import pytorch_lightning as pl

from lavis.models import load_model
from timm.models.layers import drop_path, to_3tuple, trunc_normal_
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from .eva_vit import create_eva_vit_g, PatchEmbed

from utils.utils import scaling_lr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

import loralib as lora


from sklearn.metrics import accuracy_score, roc_auc_score, r2_score



class Brain_BLIP(Blip2Base):
    def __init__(
        self,
        model_arch="blip2_t5",
        model_type="pretrain_flant5xl",
        img_size=128,
        lora_vit=False, 
        lora_llm=False,
        max_txt_len=None,
    ):
        super().__init__()
        ### setting model
        self.model = load_model(name=model_arch , model_type=model_type, is_eval=True, device='cpu')

        ### replace original 2D model's patch embedding layer and positional embedding with 3D patch embedding layer and positional embedding
        # make new 3D patch embedding layer and positional embedding
        patch_embed_3d = PatchEmbed(
            img_size=img_size, 
            #patch_size=self.model.visual_encoder.patch_embed.proj.kernel_size[0], 
            patch_size=18,
            in_chans=1, 
            embed_dim=int(self.model.visual_encoder.patch_embed.proj.out_channels))
        num_patches = patch_embed_3d.num_patches
        pos_embed_3d = nn.Parameter(torch.zeros(1, num_patches + 1, int(self.model.visual_encoder.patch_embed.proj.out_channels)))
        trunc_normal_(pos_embed_3d, std=.02)
        #pass through original 2D model's patch embedding layer and positional embedding 
        setattr(self.model.visual_encoder, "patch_embed", patch_embed_3d)
        setattr(self.model.visual_encoder,"pos_embed", pos_embed_3d)

        ### setting language encoder hyperparameters 
        self.max_txt_len = max_txt_len

        # for hugging face tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = "false"

        ### freeze parameters 
        # freeze every parameters except for patch embedding and positional embedding layer 
        for name, param in self.model.visual_encoder.named_parameters():
            if 'blocks' in name:
                param.requires_grad = False
            if 'cls_' in name: 
                param.requres_grad = False 
            if 'pos_embed' in name: 
                param.requires_grad = True 
            if 'patch_embed_' in name: 
                param.requires_grad = True
        # freeze Qformer
        for name, param in self.model.named_parameters():
            if 'Qformer' in name:
                param.requires_grad = False
            if 't5_proj' in name:
                param.requires_grad = False
        # freeze query token 
        for name, param in self.model.named_parameters():
            if 'query_tokens' in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 't5_model' in name:
                param.requires_grad = False
        for name, param in self.model.t5_model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
 


    def forward(self, batch, global_rank=None): 
        torch.cuda.empty_cache()
        #change the key name
        #batch['text_input'], batch['text_output'] = batch['inst'], batch['answer']
        #del batch['inst']
        #del batch['answer']
        loss_dict = self.model.forward(batch)
        pred = self.generate(batch)
        #pred = pred.detach().cpu().tolist()

        ### for sex classification
        #pred = [0 if sex == 'male' else 1 for sex in pred]
        ### for age classification    
        
        torch.cuda.empty_cache()
        return loss_dict['loss'], loss_dict, pred


    @torch.no_grad()
    def generate(
        self,
        batch,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=5,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        device='cuda:0',
        ):
        batch['prompt'] = batch['text_input']
        #del batch['inst']
        output_text = self.model.generate(batch)
        #print(f"GT: {batch['answer']}\nPRED:{output_text}")
        return output_text



class Brain_BLIP_pl(pl.LightningModule): 
    def __init__(self, config: dict, img_size=None):
        """
        config: dictionary load from .yaml file
        """ 
        super().__init__()
        self.save_hyperparameters(config)
        #self.automatic_optimization = False
        self.model = Brain_BLIP(model_arch=self.hparams.model.architecture,
                                model_type=self.hparams.model.type,
                                img_size=img_size,
                                lora_vit=self.hparams.model.image_encoder.lora_vit,
                                lora_llm=self.hparams.model.language_encoder.lora_llm,
                                max_txt_len=self.hparams.model.language_encoder.max_txt_len,
                                )
        

        
        # setting training hyperparameters 
        #self.learning_rate = scaling_lr(batch_size=self.hparams.training_parameters.batch_size,
        #                                accumulation_steps=self.hparams.training_parameters.accumulation_steps,
        #                                base_lr=self.hparams.training_parameters.learning_rate)
        self.learning_rate = self.hparams.training_parameters.learning_rate
        self.validation_step_outputs = None


    def summarize_model_performance(self, target, pred, ): 
        def _one_hot_encoder(text):
            """
            text	label
            This subject does not have any brain lesion.	0
            This subject has an Acute brain lesion.	1
            This subject has an Acute brain lesion characterized by hemorrhage.	2
            This subject has an Acute brain lesion characterized by ICH.	3
            This subject has an Acute brain lesion characterized by infarction.	4
            This subject has an Recent brain lesion.	5
            This subject has an Recent brain lesion characterized by hemorrhage.	6
            This subject has an Recent brain lesion characterized by ICH.	7
            This subject has an Recent brain lesion characterized by infarction.	8
            This subject has an Chronic brain lesion.	9
            This subject has an Chronic brain lesion characterized by hemorrhage.	10
            This subject has an Chronic brain lesion characterized by ICH.	11
            This subject has an Chronic brain lesion characterized by infarction.	12
            This subject has an Old brain lesion.	13
            This subject has an Old brain lesion characterized by hemorrhage.	14
            This subject has an Old brain lesion characterized by ICH.	15
            This subject has an Old brain lesion characterized by infarction.	16

            """
            """
            if "This subject does not have any brain lesion." in text:
                value = 0 
            elif "This subject has an Acute brain lesion." in text:
                value = 1
            elif "This subject has an Acute brain lesion characterized by hemorrhage." in text:
                value = 2 
            elif "This subject has an Acute brain lesion characterized by ICH." in text:
                value = 3 
            elif "This subject has an Acute brain lesion characterized by infarction." in text:
                value = 4 
            elif "This subject has an Recent brain lesion." in text:
                value = 1
            elif "This subject has an Recent brain lesion characterized by hemorrhage." in text:
                value = 2 
            elif "This subject has an Recent brain lesion characterized by ICH." in text:
                value = 3 
            elif "This subject has an Recent brain lesion characterized by infarction." in text:
                value = 4 
            elif "This subject has an Chronic brain lesion." in text:
                value = 5
            elif "This subject has an Chronic brain lesion characterized by hemorrhage." in text:
                value = 6 
            elif "This subject has an Chronic brain lesion characterized by ICH." in text:
                value = 7
            elif "This subject has an Chronic brain lesion characterized by infarction." in text:
                value = 8
            elif "This subject has an Old brain lesion." in text:
                value = 5
            elif "This subject has an Old brain lesion characterized by hemorrhage." in text:
                value = 6
            elif "This subject has an Old brain lesion characterized by ICH." in text:
                value = 7
            elif "This subject has an Old brain lesion characterized by infarction." in text:
                value = 8
            else: 
                value = -1
            """
            """
            if text in "This subject does not have any brain lesion." :
                value = 0 
            elif text in "This subject has an Acute brain lesion.":
                value = 1
            elif text in "This subject has an Recent brain lesion.":
                value = 1
            elif text in "This subject has an Chronic brain lesion.":
                value = 2
            elif text in "This subject has an Old brain lesion.":
                value = 2
            else: 
                value = -1
            """
            if "Yes" in text or "yes" in text: 
                value = 0 
            elif "No" in text or "no" in text:
                value = 1
            return value  

        assert type(target) == type(pred) == list 
        assert len(target) == len(pred)
        
        target_list = [] 
        pred_list = []
        for target_text, pred_text in zip(target, pred): 
            target_value = _one_hot_encoder(target_text)
            target_list.append(target_value)
            pred_value = _one_hot_encoder(pred_text)
            pred_list.append(pred_value)
            
            #print(f"[DEBEG] GT(Label): ({target_value}){target_text} PRED(Label): ({pred_value}){pred_text}")
        return accuracy_score(target_list, pred_list)



    def training_step(self, batch, batch_idx): 
        # reformul input data structure
        image, text, inst, answer, label = batch['image'], batch['text'], batch['inst'], batch['answer'], batch['label']
        batch['text_input'] = batch['inst']
        batch['text_output'] = batch['answer']
        loss, loss_dict, pred = self.model(batch, int(self.global_rank))
        
        try: 
            acc = self.summarize_model_performance(batch['text_output'], pred)
        except: 
            acc = -100
        
        if batch_idx % 50 == 0: 
            print(f"\nACC:{acc}\n GT: {batch['text_output'][:4]}\nPRED: {pred}")
            
        self.log_dict({
            "train/loss": loss.item(),
            'train/acc': acc
        }, sync_dist=True)
        torch.cuda.empty_cache()
        return loss
     

    def validation_step(self,batch, batch_idx): 
        # reformul input data structure
        image, text, inst, answer, label = batch['image'], batch['text'], batch['inst'], batch['answer'], batch['label']
        batch['text_input'] = batch['inst']
        batch['text_output'] = batch['answer']
        loss, loss_dict, pred = self.model(batch, int(self.global_rank))
        
        try: 
            acc = self.summarize_model_performance(batch['text_output'], pred)
        except: 
            acc = -100
            
        if batch_idx % 10 == 0: 
            print(f"\nACC:{acc}\n GT: {batch['text_output'][:4]}\nPRED: {pred}")


        self.log_dict({
            "valid/loss": loss.item(),
            'valid/acc': acc
        }, sync_dist=True)
        


    def test_step(self,batch, batch_idx): 
        # reformul input data structure
        image, text, inst, answer, label = batch['image'], batch['text'], batch['inst'], batch['answer'], batch['label']
        batch['text_input'] = batch['inst']
        batch['text_output'] = batch['answer']
        loss, loss_dict, pred = self.model(batch, int(self.global_rank))
    

        try: 
            acc = self.summarize_model_performance(batch['text_output'], pred)
        except: 
            acc = -100
        
        
        if batch_idx % 10 == 0: 
            print(f"\nACC:{acc}\n GT: {batch['text_output'][:4]}\nPRED: {pred}")

        
        self.log_dict({
            "test/loss": loss.item(),
            'test/acc': acc
        }, sync_dist=True)

    """
    def on_validation_epoch_end(self): 
        if self.validation_step_outputs is not None:
            input = self.validation_step_outputs['input']        
            inst = input['inst']
            quest = input['quest']
            answer= self.validation_step_outputs['answer']
            # make a result table and save the table
            columns = ['inst', 'quest', 'answer']
            data = list(zip(inst, quest, answer))
            self.logger.experiment.log_text(key="samples", columns=columns, data=data)
    """

    def configure_optimizers(self): 
        # setting optimizers
        if self.hparams.training_parameters.optimizer == "AdamW": 
            if self.hparams.pl_trainer.strategy == 'DeepSpeed_Zero3_offload':
                from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
                optim = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                #optim = DeepSpeedCPUAdam(self.parameters(), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                pass
            else:
                optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                #optim = torch.optim.AdamW(self.parameters(), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
            #optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
        else: 
            NotImplementedError("Only AdamW is implemented")
        
        # setting learning rate scheduler 
        if self.hparams.training_parameters.lr_scheduler == 'OneCycleLR': 
            sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": sched,
                "name": "lr_history",
                "interval": "step",
            }
            return [optim], [scheduler]
        else: 
            return optim




