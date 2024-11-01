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

class doraLeaner(myBaseLearner):
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
                    log_experiment=False):
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
                            log_experiment=log_experiment)
    
    def _load_model(self):
        base_model = super()._load_model(train_base=False)
        lora_module= LORA_MODULE_NAMES[0]
        lora_config = PeftConfig.from_pretrained(lora_module)
        dora_config=lora_config
        dora_config.use_dora=True
        dora_model = get_peft_model(base_model,dora_config)
        for name, param in base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        dora_model.print_trainable_parameters()
        return dora_model
        

                 
