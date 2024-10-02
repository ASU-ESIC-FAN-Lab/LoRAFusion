from transformers import AutoModelForSeq2SeqLM
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy
import random
import nevergrad as ng
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict
from peft import PeftModel, PeftConfig,get_peft_model
from functools import partial
from typing import List, Optional, Union
import copy
from torch.autograd import Variable
import torch.optim as optim
import gc  # Garbage collection
from torchviz import make_dot
import os
import time
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: Optional[str] = None):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # print(default_peft_model_id)
    #dora config
    dora_config = PeftConfig.from_pretrained(default_peft_model_id)
    dora_config.use_dora = True
    dora_config.lora_dropout = 0.3
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
        
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 0 is the default model
    try:
        #create a new base model with dora config
        dora_base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        peft_model = get_peft_model(dora_base_model, dora_config)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
    base_model = base_model.to(device)
    peft_model = peft_model.to(device)
    peft_model.eval()

    cache = {}
    return peft_model, tokenizer, cache

def preprocess_function(examples, tokenizer):
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


def load_dataset(example_inputs, example_outputs, tokenizer):
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_loss(example_dataset, model, batch_size):
    """
    Get the loss of the model on the example dataset.
    If compute_gradients is True, computes the loss with gradients (for training).
    Otherwise, computes the loss without gradients (for evaluation).
    """
    data_batch_size = len(example_dataset) if batch_size is None else min(len(example_dataset), batch_size)
    
    # Create a DataLoader to batch the example dataset
    train_dataloader = DataLoader(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,  # If True, the data loader will copy tensors into CUDA pinned memory before returning them
    )
    
    train_loss = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.config.use_cache = False 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _, batch in enumerate(train_dataloader):
        input("before train loader Press Enter to continue...")
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the appropriate device
        print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device)} bytes")
        # optimizer.zero_grad()  # Clear gradients before the forward pass
        # Compute loss with gradients (training)
        outputs = model(**batch)  # Forward pass through the model
        print(f"Memory allocated after output: {torch.cuda.memory_allocated(device)} bytes")

        loss = outputs.loss
        loss.backward()  # Backpropagate the loss
        print(f"Memory allocated after backward: {torch.cuda.memory_allocated(device)} bytes")
        # optimizer.step()  # Update the model's parameters
        # print(f"Memory allocated after step: {torch.cuda.memory_allocated(device)} bytes")
        train_loss += loss  # Accumulate the loss
        del batch, outputs, loss  # Clear memory
        model.zero_grad() 
        torch.cuda.empty_cache()  # Clear memory
        
        gc.collect()  # Garbage collection
        print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
    
    return train_loss / len(example_dataset["input"])  # Keep as tensor for backpropagation

def default_l1_regularization(weights):
    """
    Get the L1 regularization term for the weights
    """
    sum_of_squares = sum([abs(x) for x in weights.flatten()]) / len(weights.flatten())
    return 0.0005 * sum_of_squares

def get_score(weights, model, cache, example_dataset, batch_size, get_loss, get_regular):

    # minimize the metric
    loss = get_loss(example_dataset, model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)

    # metric_val = loss
    
    return metric_val

def get_final_weights(weights, lora_module_list, cache):
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for j,key in enumerate(keys):
                if 'encoder' in key:
                    final_state_dict[key] = weights[i][j//4] * lora_state_dict[key]
                else:
                    final_state_dict[key] = weights[i][j//8+12] * lora_state_dict[key]
        else:
            for j,key in enumerate(keys):
                if 'encoder' in key:
                    final_state_dict[key] = (
                        final_state_dict[key] + weights[i][j//4] * lora_state_dict[key]
                    )
                else:
                    final_state_dict[key] = (
                        final_state_dict[key] + weights[i][j//8+12] * lora_state_dict[key]
                    )

    return final_state_dict
    
def lorahub_inference(example_inputs: List[str],
                      model_or_name_path: Union[AutoModelForSeq2SeqLM, str],
                      tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
                      batch_size: int,
                      # if not provided, we do not report the accuracy
                      example_outputs: List[str]=None):
    
    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if output.strip().lower().replace(".", "") == truth.strip().lower().replace(".", ""):
                correct += 1
            total += 1
        return correct / total * 100

    example_predictions = []
    # load model
    if isinstance(model_or_name_path, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_or_name_path)
    else:
        model = model_or_name_path
    
    # load tokenizer
    if isinstance(tokenizer_or_tokenizer_path, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_path)
    else:
        tokenizer = tokenizer_or_tokenizer_path
            
    # for name, param in model.named_parameters():
    #     if "lora" not in name:
    #         param.requires_grad = False
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"], max_new_tokens=256
        )
        outputs = tokenizer.batch_decode(
            outputs.to("cpu"), skip_special_tokens=True
        )
        example_predictions.extend(outputs)
    
    if example_outputs is not None:
        task_perf = accuracy_score(example_predictions, example_outputs)
    else:
        task_perf = None
        
    return example_predictions, task_perf


def lorahub_learning(lora_module_list: List[str], 
                     example_inputs: List[str], 
                     example_outputs: List[str], 
                     max_inference_step: int,
                     model_name_or_path=None,
                     batch_size=None,
                     get_loss=default_get_loss, 
                     get_regular=default_l1_regularization,
                     seed=42,
                     early_stopping=True,
                     lr=0.01,
                     valid_inputs=None,
                     valid_outputs=None):
    # set seed for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None

    # load model
    model, tokenizer, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path)

    dataset = load_dataset(example_inputs, example_outputs, tokenizer) 
    data_batch_size = len(dataset) if batch_size is None else min(len(dataset), batch_size)
    train_dataloader = DataLoader(
        dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,  # If True, the data loader will copy tensors into CUDA pinned memory before returning them
    )
    if valid_inputs is not None:
        valid_dataset = load_dataset(valid_inputs, valid_outputs, tokenizer)    
        valid_dataloader = DataLoader(
            valid_dataset,
            collate_fn=default_data_collator,
            batch_size=data_batch_size,
            pin_memory=True,  # If True, the data loader will copy tensors into CUDA pinned memory before returning them
        )
    params_list=[]
    for name, param in model.named_parameters():
        if "lora" in name:
        # if "lora_magnitude_vector" in name:
            param.requires_grad = True
            params_list.append(param)
    print(len(params_list))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = list[filter(lambda p: p.requires_grad, model.parameters())]
    print(len(list(params)))
    # torch.nn.init.xavier_uniform_(params)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=0.001)

    patience = 10
    warmup_steps = 5
    best_loss = float("inf")
    best_params = None
    patience_counter = 0
    for step in range(max_inference_step):
        total_loss = 0
        
        for _, batch in enumerate(train_dataloader):
            # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss/len(batch["input_ids"])
            # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            del batch, outputs, loss  # Clear memory
            # print("update time:",time.time()-starttime)
        avg_train_loss = total_loss / len(train_dataloader)
        if valid_inputs is not None:
            
            with torch.no_grad():
                total_loss = 0
                for _, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss/len(batch["input_ids"])
                    total_loss += loss.item()
                    del batch, outputs, loss
            avg_valid_loss = total_loss / len(valid_dataloader)
            if step % 1 == 0:
                print(f"Step {step}, train loss {avg_train_loss}, valid loss {avg_valid_loss}")
            if early_stopping:
                if avg_valid_loss < best_loss:
                    best_loss = avg_valid_loss
                    # best_params = params.detach().clone()
                    best_params = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    if step > warmup_steps:
                        patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at step {step}")
                        model.load_state_dict(best_params)
                        break
        else:
            print(f"Step {step}, train loss {avg_train_loss}")


    # set the final weights
    # set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    del optimizer,train_dataloader,dataset
    # del  optimizer, final_state_dict,final_lora,train_dataloader,dataset
    # del key_params_lookup, model_param_name_lookup,optimized_weights,
    # print("test")
    return None, model, tokenizer