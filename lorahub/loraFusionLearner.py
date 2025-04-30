from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer,BitsAndBytesConfig
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig,get_peft_model
from functools import partial
from typing import List, Optional, Union
import copy
from torch.autograd import Variable
import torch.optim as optim
from lorahub.constant import LORA_MODULE_NAMES
import os
import time
import wandb
from lorahub.baseLearner import myBaseLearner
import bitsandbytes as bnb
from lorahub.myQuantTensor4bit import myQuantTensor4bit

class loraFusionLearner(myBaseLearner):
    def __init__(self, model_name_or_path="google/flan-t5-large", 
                    batch_size=5,
                    seed=42,
                    lr=1e-4,
                    max_step=20,
                    train_input=None,
                    train_output=None,
                    valid_input=None,
                    valid_output=None,
                    prune=False,
                    early_stopping=False,
                    load_in_4bit=False,
                    load_in_8bit=False,
                    log_experiment=False,
                    lora_num=20,
                    **kwargs):
        self.lora_num = lora_num
        self.lora_module_list = None
        super().__init__(model_name_or_path=model_name_or_path,
                            batch_size=batch_size,
                            seed=seed,
                            lr=lr,
                            max_step=max_step,
                            train_input=train_input,
                            train_output=train_output,
                            valid_input=valid_input,
                            valid_output=valid_output,
                            prune=prune,
                            early_stopping=early_stopping,
                            load_in_4bit=load_in_4bit,
                            load_in_8bit=load_in_8bit,
                            log_experiment=log_experiment,
                            **kwargs)
    
    def _load_model(self):
        base_model = super()._load_model(train_base=False)
        lora_module= LORA_MODULE_NAMES[0]
        lora_config = PeftConfig.from_pretrained(lora_module)

        lora_model = get_peft_model(base_model,lora_config)
        for name, param in base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        #save the initial state dict
        
        # lora_model.print_trainable_parameters()
        tmp_model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name,device_map=self.device)
        #get random lora modules
        lora_module_list = random.sample(LORA_MODULE_NAMES, self.lora_num)
        self.lora_module_list = lora_module_list
        self.lora_dict_caches = {}
        first_dict = None
        for peft_model_id in tqdm(lora_module_list):
            # print("> Loading {} ...".format(peft_model_id))
            cur_peft_model = PeftModel.from_pretrained(tmp_model, peft_model_id)
            self.lora_dict_caches[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))
            #quantize the lora models
            if self.quantization_config is not None:
                for name, param in self.lora_dict_caches[peft_model_id].items():
                    # print(param)
                    param = myQuantTensor4bit(param)

            if first_dict is None:
                first_dict = self.lora_dict_caches[peft_model_id]
            # check whether the LoRA can be merged into one 
            try:
                # detect whether the arch is the same
                for key in first_dict.keys():
                    assert first_dict[key].shape == self.lora_dict_caches[peft_model_id][key].shape
            except:
                raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
        del tmp_model
        
        return lora_model
    @staticmethod
    def merge_lora_module(weights, lora_module_list, cache):
        merged_state_dict = {}
        keys = cache[lora_module_list[0]].keys()
        for i, peft_model_id in enumerate(lora_module_list):
            lora_state_dict = cache[peft_model_id]
            if i == 0:
                for j,key in enumerate(keys):
                    if 'encoder' in key:
                        
                        merged_state_dict[key] = weights[i][j//4] * lora_state_dict[key]
                    else:
                        merged_state_dict[key] = weights[i][j//8+12] * lora_state_dict[key]
            else:
                for j,key in enumerate(keys):
                    if 'encoder' in key:
                        merged_state_dict[key] = (
                            merged_state_dict[key] + weights[i][j//4] * lora_state_dict[key]
                        )
                    else:
                        merged_state_dict[key] = (
                            merged_state_dict[key] + weights[i][j//8+12] * lora_state_dict[key]
                        )

        return merged_state_dict   
     
    def train(self):
        num_blocks=len(self.lora_dict_caches[self.lora_module_list[0]].keys())//6
        # print("num_blocks:",num_blocks)
        
        params_magnitude = []
        #create a dummpy lora state dict for fusion of lora modules
        merged_state_dict = {}
        keys = self.lora_dict_caches[self.lora_module_list[0]].keys()
        key_params_lookup = {} #lora parameter name to the corresponding lora weights in different lora modules
        model_param_name_lookup={}#lookup table for name and param of the base model

        #set requires_grad to True for lora parameters
        for name, param in self.model.named_parameters():
            # print(name)
            if "lora" in name:
                name_processed = name.replace(".default","")
                if name_processed in self.lora_dict_caches[self.lora_module_list[0]]:
                    model_param_name_lookup[name_processed]=param
                    param.requires_grad = True

        #trainable parameter and optimizers
        params = torch.empty(self.lora_num, num_blocks, device=self.device, requires_grad=True)
        torch.nn.init.xavier_uniform_(params)
        if self.quantization_config is not None:
            optimizer_direction = bnb.optim.Adam([params],lr=self.lr,weight_decay=0.0000, percentile_clipping=95)
        else:
            optimizer_direction = optim.Adam([params], lr=self.lr,weight_decay=0.00001)

        def update_lora():
            for i, peft_model_id in enumerate(self.lora_module_list):
                lora_state_dict = self.lora_dict_caches[peft_model_id]
                if i == 0:
                    for j,key in enumerate(keys):
                        if 'encoder' in key:
                            # print(merged_state_dict[key].is_cuda)
                            merged_state_dict[key] = params[i][j//4] * lora_state_dict[key]
                            key_params_lookup[key] = [(i,j//4,peft_model_id)]
                        
                        else:
                            merged_state_dict[key] = params[i][j//8+12] * lora_state_dict[key]
                            key_params_lookup[key] = [(i,j//8+12,peft_model_id)]
                else:
                    for j,key in enumerate(keys):
                        if 'encoder' in key:
                            merged_state_dict[key] = (
                                merged_state_dict[key] + params[i][j//4] * lora_state_dict[key]
                            )
                            key_params_lookup[key].append((i,j//4,peft_model_id))
                        else:
                            merged_state_dict[key] = (
                                merged_state_dict[key] + params[i][j//8+12] * lora_state_dict[key]
                            )
                            key_params_lookup[key].append((i,j//8+12,peft_model_id))
            for name,param in model_param_name_lookup.items():
                param.data.copy_(merged_state_dict[name])
                # print(param.data)
                param.grad = None
        
        update_lora()
        # print("> Begin to perform gradient optimization ...")
        patience = self.max_step//3
        warmup_steps = self.max_step//5
        patience_counter = 0
        best_loss = float("inf")
        best_params = None
        # check_nan_in_parameters(model)
        for step in range(self.max_step):
            total_loss = 0
            
            for _, batch in enumerate(self.train_dataloader):
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
                optimizer_direction.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                # print(outputs)
                loss = outputs.loss/len(batch["input_ids"])
                # print(loss)
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
                total_loss += loss.item()
                loss.backward()
                
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
                l1reg=self.default_l1_regularization(params)
                l1reg.backward()
                self.check_nan_in_gradients(self.model)
                for name, param in model_param_name_lookup.items():
                    for i,j,peft_model_id in key_params_lookup[name]:
                        # print(name,params[i,j])
                        params.grad[i,j] += 2* params.data[i,j] * (param.grad * self.lora_dict_caches[peft_model_id][name]).sum()

                optimizer_direction.step()
                # nan=check_nan_in_parameters(model)
                update_lora()
                del batch, outputs, loss  # Clear memory
                # print("update time:",time.time()-starttime)
                # print("test")
            avg_train_loss = total_loss / len(self.train_dataloader) + l1reg.item()
            #valid loss
            _, valid_acc = self.inference(dataset=self.valid_dataset)

            _, train_acc = self.inference(dataset=self.train_dataset)
                
            
            if self.valid_dataloader is not None:
                
                with torch.no_grad():
                    total_loss = 0
                    for _, batch in enumerate(self.valid_dataloader):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        loss = outputs.loss/len(batch["input_ids"])
                        total_loss += loss.item()
                        del batch, outputs, loss
                avg_valid_loss = total_loss / len(self.valid_dataloader)
                if step % 1 == 0:
                    # result
                    print(f"Step {step}, train loss {avg_train_loss}, train acc {train_acc}, valid loss {avg_valid_loss}, valid acc {valid_acc}")
                    if self.log_experiment:
                        wandb.log({"train_loss":avg_train_loss,"train_acc":train_acc,"valid_loss":avg_valid_loss,"valid_acc":valid_acc})
                
                    # print(f"Step {step}, train loss {avg_train_loss}, valid loss {avg_valid_loss}")
                if self.early_stopping:
                    if avg_valid_loss < best_loss:
                        best_loss = avg_valid_loss
                        # best_params = params.detach().clone()
                        best_params = params.detach().clone()
                        patience_counter = 0
                    else:
                        if step > warmup_steps:
                            patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at step {step}")
                            break
                else:
                    best_params = params.detach().clone()
            else:
                print(f"Step {step}, train loss {avg_train_loss}")
            
            #calculate mask based on the asb value of the magnitude vector, prune the 15% smallest value



        
        
        optimized_weights = best_params.cpu().numpy()
        final_lora = self.merge_lora_module(optimized_weights, self.lora_module_list, self.lora_dict_caches)
        # set the final weights
        set_peft_model_state_dict(self.model, final_lora)
        # self.model = self.model.merge_and_unload()
        # del params, optimizer,magnitude_optimizer, merged_state_dict,final_lora,train_dataloader,dataset
        # del key_params_lookup, model_param_name_lookup,optimized_weights,

                 
