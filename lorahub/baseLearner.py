from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer,BitsAndBytesConfig
import pandas as pd
import numpy as np
import random
from functools import partial
from typing import List, Optional, Union
import torch.optim as optim
import wandb
import bitsandbytes as bnb
from tqdm import tqdm

class myBaseLearner:
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
                    **kwargs):
        self.lr=lr
        self.seed=seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.max_step=max_step
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.base_model_name = model_name_or_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_experiment = log_experiment
        self.tokenizer = None
        self.quantization_config = None
        if load_in_4bit:
            print("loading in 4bit")
            self.quantization_config= BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
        elif load_in_8bit:
            print("loading in 8bit")
            self.quantization_config= BitsAndBytesConfig(load_in_8bit=True)

        self.model = None
        self.model=self._load_model()
        self.train_dataset = None
        self.valid_dataset = None
        self.train_dataloader = None
        self.valid_dataloader = None
        
        self.train_dataset = self.load_dataset(train_input, train_output)
        self.valid_dataset = self.load_dataset(valid_input, valid_output)
        self.train_dataloader = self.load_data_loader(self.train_dataset)
        self.valid_dataloader = self.load_data_loader(self.valid_dataset)

        if load_in_4bit and load_in_8bit:
            raise ValueError("Cannot load in both 4-bit and 8-bit")
        


        if model_name_or_path is None:
            raise ValueError("model_name_or_path is required")

    def _load_model(self,train_base=True):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name,quantization_config=self.quantization_config,device_map='auto')
        # if self.quantization_config is not None:
        #     base_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if train_base:
            base_model.train()
        # self.model=base_model
        return base_model
    @staticmethod
    def _preprocess_function(examples,tokenizer):
        """
        standard preprocess function for dataset
        """
        inputs = examples["input"]
        targets = examples["output"]
        model_inputs = tokenizer(
            inputs,
            max_length=2048,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            targets,
            max_length=2048,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs
    @staticmethod
    def check_nan_in_gradients(model):
        for name, param in model.named_parameters():
            if param.grad is not None:  # Check if the parameter has a gradient
                if torch.isnan(param.grad).any():
                    #set to 0
                    param.grad = torch.zeros_like(param.grad)
                    print(f"NaN detected in gradient of parameter: {name}")
                    # return True
                if torch.isinf(param.grad).any():
                    #set to 0
                    param.grad = torch.zeros_like(param.grad)
                    print(f"Infinity detected in gradient of parameter: {name}")
                    # return True
        return False
    @staticmethod
    def check_nan_in_parameters(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
                return True
        print("No NaNs detected in parameters.")
        return False
    def load_dataset(self,example_inputs, example_outputs):
        # add empty string if example_outputs is None
        if example_outputs is None:
            return None
        df = [
            {"input": example_inputs[i], "output": example_outputs[i]}
            for i in range(len(example_inputs))
        ]
        dataset = Dataset.from_pandas(pd.DataFrame(df))
        preprocess_func_with_tokenizer = partial(self._preprocess_function,tokenizer=self.tokenizer)
        processed_datasets = dataset.map(
            preprocess_func_with_tokenizer,
            batched=True,
            num_proc=1,
            desc="Running tokenizer on dataset",
        )
        return processed_datasets
    def load_data_loader(self,dataset):
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=default_data_collator,
            pin_memory=True,
        )
    
    @staticmethod
    def default_l1_regularization(weights):
        """
        Get the L1 regularization term for the weights
        """
        sum_of_squares = sum([abs(x) for x in weights.flatten()]) / len(weights.flatten())
        return 0.0005 * sum_of_squares
     
    
    def inference(self,example_inputs: List[str]=None,
                      # if not provided, we do not report the accuracy
                      example_outputs: List[str]=None,
                      dataset=None):
        def accuracy_score(outputs, ground_truths):
            correct = 0
            total = 0
            for output, truth in zip(outputs, ground_truths):
                if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                    correct += 1
                total += 1
            return correct / total * 100

        example_predictions = []
        if not dataset:
            dataset = self.load_dataset(example_inputs, example_outputs)
        # use gpu if available
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # for i in range(0, len(dataset["input"]), self.batch_size):
        for i in tqdm(range(0, len(dataset["input"]), self.batch_size)):
            inputs = self.tokenizer(
                dataset["input"][i : i + self.batch_size],
                # max_length=2048,
                return_tensors="pt",
                padding=True,
            ).to(device)
            outputs = self.model.generate(
                input_ids=inputs["input_ids"], max_new_tokens=256
            )
            outputs = self.tokenizer.batch_decode(
                outputs.to("cpu"), skip_special_tokens=True
            )
            example_predictions.extend(outputs)
        
        example_outputs=dataset["output"]
        if example_outputs is not None:
            task_perf = accuracy_score(example_predictions, example_outputs)
        else:
            task_perf = None
        
        return example_predictions, task_perf
    def save_model(self,save_path):
        self.model.save_pretrained(save_path)
    
    def train(self,validation=True):
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is required")
        print("start training")
        validation = validation and self.valid_dataloader is not None
        if self.quantization_config is not None:
            optimizer = bnb.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                   lr=self.lr, optim_bits=32, percentile_clipping=95,weight_decay=0.00001)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr,weight_decay=0.00001)
        # optimizer = bnb.optim.Adam8bit(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        #                             , betas=(0.9, 0.995), optim_bits=32, percentile_clipping=5)

        for step in range(self.max_step):
            total_loss = 0

            for _,batch in enumerate(self.train_dataloader):
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(self.device)} bytes")
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss/len(batch["input_ids"])
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(self.device)} bytes")
                # input("press any key to continue")
                total_loss += loss.item()
                loss.backward()
                self.check_nan_in_gradients(self.model)
                optimizer.step()
                del batch, outputs, loss  # Clear memory
                
                # print("update time:",time.time()-starttime)
            avg_train_loss = total_loss / len(self.train_dataloader)

            _,train_acc = self.inference(dataset=self.train_dataset)
            _,valid_acc = self.inference(dataset=self.valid_dataset)

            if validation:
                with torch.no_grad():
                    total_loss = 0
                    for _,batch in enumerate(self.valid_dataloader):
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        loss = outputs.loss/len(batch["input_ids"])
                        total_loss += loss.item()
                        del batch, outputs, loss
                avg_valid_loss = total_loss / len(self.valid_dataloader)
            if step % 1 == 0:
                if validation:
                    print(f"step {step}, loss {avg_train_loss}, train acc {train_acc}, valid loss {avg_valid_loss}, valid acc {valid_acc}")
                    if self.log_experiment:
                        wandb.log({"train_loss":avg_train_loss,"train_acc":train_acc,"valid_loss":avg_valid_loss,"valid_acc":valid_acc})
                else:
                    print(f"step {step}, loss {avg_train_loss}, train acc {train_acc}")
                    if self.log_experiment:
                        wandb.log({"train_loss":avg_train_loss,"train_acc":train_acc})
            # if early_stopping and validation:
            #     if step > 1 and avg_valid_loss > prev_valid_loss:
            #         break
            #     prev_valid_loss = avg_valid_loss

        del optimizer
        

