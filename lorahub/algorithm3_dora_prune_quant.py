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
import gc  # Garbage collection
from torchviz import make_dot
import os
import time
import wandb
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def load_base_model_and_lora_modules(lora_module_list: List[str], model_name_or_path: Optional[str] = None,quantization_config=None):
    """load base model and lora modules from huggingface model hub

    Args:
        lora_module_list (List[str]): a list of lora module names available in huggingface model hub
        model_name_or_path (Optional[str]): base model name, default is None
    """
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]
    #dora config
    dora_config = PeftConfig.from_pretrained(default_peft_model_id)
    dora_config.use_dora = True
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(default_peft_model_id).base_model_name_or_path
        
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 0 is the default model
    try:
        #create a new base model with dora config
        if quantization_config is not None:
            dora_base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,quantization_config=quantization_config)
        else:
            dora_base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        peft_model = get_peft_model(dora_base_model, dora_config)
    except:
        raise Exception(f'{default_peft_model_id} is unable to load into the model {model_name_or_path}')
    base_model = base_model.to(device)
    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        # print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = copy.deepcopy(get_peft_model_state_dict(cur_peft_model))

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one 
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(f'LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank).')
               
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
    if not dataset:
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
                     get_loss=None, 
                     get_regular=default_l1_regularization,
                     seed=42,
                     early_stopping=True,
                     lr=0.01,
                     valid_inputs=None,
                     valid_outputs=None,
                     log_experiment=False,
                     prune=True,
                     load_in_4bit=False,
                     load_in_8bit=False):
    print("init dora learning")
    # set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print("> No LoRA modules are provided. Please provide at least one LoRA module.")
        return None, None
    if load_in_4bit and load_in_8bit:
        print("> Please provide only one quantization method.")
        return None, None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16)
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    # load model
    model, tokenizer, cache = load_base_model_and_lora_modules(lora_module_list, model_name_or_path,quantization_config=quantization_config)
    # for name,param in model.named_parameters():
    #     print(name,param.dtype)


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

    
    
    # set up the limit of the weights dimension 24 * number_of_loras
    num_blocks=len(cache[lora_module_list[0]].keys())//6
    print("num_blocks:",num_blocks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = torch.empty(number_of_loras, num_blocks, device=device, requires_grad=True)
    torch.nn.init.xavier_uniform_(params)
    optimizer = optim.Adam([params], lr=lr, weight_decay=0.00000) #weight decay
    magnitude_params = []
    
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    key_params_lookup = {} #parameter name to the corresponding lora weights for different modules
    model_param_name_lookup={}#parameter name in the model to correspond parameter in model 
    for name, param in model.named_parameters():
        # print(name)
        if "lora" in name:
            name_processed = name.replace(".default","")
            if name_processed in cache[lora_module_list[0]]:
                model_param_name_lookup[name_processed]=param
                param.requires_grad = True
        if "lora_magnitude_vector" in name:
            # print(name)
            param.requires_grad = True
            magnitude_params.append(param)
    # print(len(magnitude_params))
    #create a empty dummy tensor for each parameter
    magnitude_params2 = [torch.ones(len(magnitude_params), device=device, requires_grad=True)]
    magnitude_optimizer = optim.Adam(magnitude_params, lr=0.005)
    magnitude_mask = torch.ones(len(magnitude_params), device=device, requires_grad=False)

    def check_nan_in_gradients(model):
        for name, param in model.named_parameters():
            if param.grad is not None:  # Check if the parameter has a gradient
                if torch.isnan(param.grad).any():
                    #set to 0
                    param.grad = torch.zeros_like(param.grad)
                    # print(f"NaN detected in gradient of parameter: {name}")
                    # return True
                if torch.isinf(param.grad).any():
                    #set to 0
                    param.grad = torch.zeros_like(param.grad)
                    # print(f"Infinity detected in gradient of parameter: {name}")
                    # return True
        return False
    def check_nan_in_parameters(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
                return True
        print("No NaNs detected in parameters.")
        return False
    def update_lora():
        for i, peft_model_id in enumerate(lora_module_list):
            lora_state_dict = cache[peft_model_id]
            if i == 0:
                for j,key in enumerate(keys):
                    if 'encoder' in key:
                        # print(final_state_dict[key].is_cuda)
                        final_state_dict[key] = params[i][j//4] * lora_state_dict[key]
                        key_params_lookup[key] = [(i,j//4,peft_model_id)]
                    
                    else:
                        final_state_dict[key] = params[i][j//8+12] * lora_state_dict[key]
                        key_params_lookup[key] = [(i,j//8+12,peft_model_id)]
            else:
                for j,key in enumerate(keys):
                    if 'encoder' in key:
                        final_state_dict[key] = (
                            final_state_dict[key] + params[i][j//4] * lora_state_dict[key]
                        )
                        key_params_lookup[key].append((i,j//4,peft_model_id))
                    else:
                        final_state_dict[key] = (
                            final_state_dict[key] + params[i][j//8+12] * lora_state_dict[key]
                        )
                        key_params_lookup[key].append((i,j//8+12,peft_model_id))
        for name,param in model_param_name_lookup.items():
            param.data.copy_(final_state_dict[name])
            # print(param.data)
            param.grad = None
    
        # for name, param in model.named_parameters():
        #     #extract string .default from the name
        #     name_processed = name.replace(".default","")
        #     if name_processed in final_state_dict:
        #         # print(f"{name} requires gradient.")
                
        #         param.data.copy_(final_state_dict[name_processed])
        #         param.requires_grad = True
        #         param.grad = None
    update_lora()
    # print("> Begin to perform gradient optimization ...")
    patience = max_inference_step//3
    warmup_steps = max_inference_step//5
    patience_counter = 5
    prune_step = 10
    best_loss = float("inf")
    best_params = None
    # check_nan_in_parameters(model)
    for step in range(max_inference_step):
        total_loss = 0
        
        for _, batch in enumerate(train_dataloader):
            # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
            optimizer.zero_grad()
            magnitude_optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # print(outputs)
            loss = outputs.loss/len(batch["input_ids"])
            # print(loss)
            # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
            total_loss += loss.item()
            loss.backward()
            
            # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(device)} bytes")
            l1reg=default_l1_regularization(params)
            l1reg.backward()

            # for name, param in model.named_parameters():
            #     if name == 'base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.default.weight':
            #         print(param.data)
            #     if param.grad is not None:  # Check if the parameter has a gradient
            #         #print min and max of the gradient
            #         # print(name,param.data.max())
            #         pass
            # # print(magnitude_params[0].data)
            
            gradient=check_nan_in_gradients(model)
            #apply mask
            if prune:
                for i,param in enumerate(magnitude_params):
                    param.data *= magnitude_mask[i]
                    #set grad to zero not None
                    if magnitude_mask[i] == 0:
                        param.grad = torch.zeros_like(param.data)
                    # param.grad = torch.zeros_like(param.data)
            for name, param in model_param_name_lookup.items():
                for i,j,peft_model_id in key_params_lookup[name]:
                    # print(name,param.grad)
                    params.grad[i,j] += (param.grad * cache[peft_model_id][name]).sum()
            # torch.nn.utils.clip_grad_norm_(params, 1.0) 
            # torch.nn.utils.clip_grad_norm_(magnitude_params, 1.0)
            optimizer.step()
            magnitude_optimizer.step()
            # nan=check_nan_in_parameters(model)
            update_lora()
            del batch, outputs, loss  # Clear memory
            # print("update time:",time.time()-starttime)
        avg_train_loss = total_loss / len(train_dataloader) + l1reg.item()
        #valid loss
        _, valid_acc = lorahub_inference(valid_inputs, model, tokenizer, batch_size, valid_outputs,dataset=valid_dataset)

        _,train_acc = lorahub_inference(example_inputs, model, tokenizer, batch_size, example_outputs,dataset=dataset)
            
        
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
                # result
                print(f"Step {step}, train loss {avg_train_loss}, train acc {train_acc}, valid loss {avg_valid_loss}, valid acc {valid_acc}")
                if log_experiment:
                    wandb.log({"train_loss":avg_train_loss,"train_acc":train_acc,"valid_loss":avg_valid_loss,"valid_acc":valid_acc})
            
                # print(f"Step {step}, train loss {avg_train_loss}, valid loss {avg_valid_loss}")
            if early_stopping:
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
        if prune:
            with torch.no_grad():
                prune_rate = 0.15
                if step == prune_step:
                    print("prune")
                    magnitude = torch.cat([param.data for param in magnitude_params])
                    magnitude_abs = torch.abs(magnitude)
                    #save the magnitude distribution
                    magnitude_distribution = magnitude_abs.cpu().numpy()
                    # np.save("magnitude_distribution.npy",magnitude_distribution)
                    #sort
                    sorted_magnitude, sorted_indices = torch.sort(magnitude_abs)
                    threshold = sorted_magnitude[int(len(sorted_magnitude)*prune_rate)]
                    magnitude_mask = (magnitude_abs >= threshold).float()
                #apply mask
                for i,param in enumerate(magnitude_params):
                    param.data *= magnitude_mask[i]


        

        
    
    optimized_weights = best_params.cpu().numpy()
    #save value to txt
    # with open('recommendation.txt', 'w') as f:
    #     for item in optimized_weights:
    #         f.write("%s\n" % item)
    final_lora = get_final_weights(optimized_weights, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model, final_lora)
    # for name,param in model.named_parameters():
    #     print(name,param.dtype)
    model = model.merge_and_unload()
    # for name,param in model.named_parameters():
    #     print(name,param.dtype)
    # exit()
    del params, optimizer,magnitude_optimizer, final_state_dict,final_lora,train_dataloader,dataset
    del key_params_lookup, model_param_name_lookup,optimized_weights,
    # print("test")
    return None, model, tokenizer