from lorahub.algorithm import lorahub_inference
import os
import json
from lorahub.algorithm3_dora import lorahub_learning, lorahub_inference
from lorahub.constant import LORA_MODULE_NAMES
import random
from random import shuffle
import sys
import pandas as pd
import torch
def evaluate_flan_results_zero_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    result={}
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (zero shot): ", sub_dir)
        _,task_acc=lorahub_inference(task_inputs,
                          flan_model_name,
                          flan_model_name,
                          16,
                          task_outputs)
        print("task accuracy:",task_acc)
        result[sub_dir]=task_acc
    result_pd=pd.DataFrame({'Zero-shot acc':result})
    return result,result_pd
        # break


def evaluate_flan_results_few_shot(folder, flan_model_name):
    sub_dirs = os.listdir(folder)
    result={}
    for sub_dir in sub_dirs:
        test_file_path = os.path.join(folder, sub_dir, "few_shot.jsonl")
        task_inputs, task_outputs = [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
        print("Evaluating on task (five shot): ", sub_dir)
        _,task_acc=lorahub_inference(task_inputs,
                          flan_model_name,
                          flan_model_name,
                          16,
                          task_outputs)
        result[sub_dir]=task_acc
        print("task accuracy:",task_acc)
        # break
    result_pd=pd.DataFrame({'Few-shot acc':result})
    return result,result_pd

def separate_valid_dataset(example_inputs, examples_outputs, valid_ratio=0.1):
    example_num = len(example_inputs)
    valid_num = int(example_num * valid_ratio)
    valid_inputs, valid_outputs = example_inputs[:valid_num], examples_outputs[:valid_num]
    train_inputs, train_outputs = example_inputs[valid_num:], examples_outputs[valid_num:]
    return train_inputs, train_outputs, valid_inputs, valid_outputs
def evaluate_lorahub_results_few_shot(folder, flan_model_name,save_path="results"):
    sub_dirs = os.listdir(folder)
    sub_dirs= sorted(sub_dirs)
    result={}
    # result={'lorahub avg acc':{},'lorahub max acc':{}}
    # 5 seeds used in our experiments
    for sub_dir in sub_dirs:
        # try:
            # if "boolean_expression" in sub_dir:
            #     continue
            print("Evaluating on task (lorahub): ", sub_dir)
            # construct the few-shot examples for lorahub learning
            example_inputs, examples_outputs = [], []
            example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
            for line in open(example_file_path, "r", encoding="utf-8"):
                example = json.loads(line)
                example_inputs.append(example["context"])
                examples_outputs.append(example["completion"])
                
            # random select 5 examples for each task
            random.seed(42)
            shuffled_set = list(zip(example_inputs, examples_outputs))
            random.shuffle(shuffled_set)
            example_inputs, examples_outputs = zip(*shuffled_set)
            # take the first 5 examples
            example_num=100
            example_inputs, examples_outputs = example_inputs[:example_num], examples_outputs[:example_num]
            # separate the training and validation dataset
            train_inputs, train_outputs, valid_inputs, valid_outputs = separate_valid_dataset(example_inputs, examples_outputs, valid_ratio=0.15)

            # load the zero-shot examples for evaluation
            test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
            task_inputs, task_outputs = [], []
            for line in open(test_file_path, "r", encoding="utf-8"):
                example = json.loads(line)
                task_inputs.append(example["context"])
                task_outputs.append(example["completion"])

            step_result={}
            for step in range(20,21,1):
                for lora_num in range(20,21,1):
                    for lr_n in range(20,21,5):
                        lr=lr_n/1000
                        print(lr)
                        task_perf_list = []
                        if (step,lora_num,lr_n) not in result.keys():
                            result[(step,lora_num,lr_n)]={'lorahub avg acc':{},'lorahub max acc':{}}

                        for seed in range(1,4):
                            
                            # lr=0.001
                            random.seed(seed)
                            print("Evaluating on task (lorahub): ", sub_dir, "with seed:", seed)
            
                            def get_lora_module_list(lora_num=40):
                                return random.sample(LORA_MODULE_NAMES, lora_num) #what 
                            # get a list of modules to be used in the composition
                            modules = get_lora_module_list(lora_num)

                            # perform LoRAHub learning
                            module_weights, model, tokenizer = lorahub_learning(lora_module_list=modules,
                                                                                example_inputs=train_inputs,
                                                                                example_outputs=train_outputs,
                                                                                max_inference_step=step,
                                                                                batch_size=5,lr=lr,
                                                                                valid_inputs=valid_inputs,
                                                                                valid_outputs=valid_outputs)

                            # print("module_weights:", module_weights)

                            """
                            Perform inference to get predictions
                            """
                            _, task_acc = lorahub_inference(example_inputs=task_inputs,
                                                            model_or_name_path=model,
                                                            tokenizer_or_tokenizer_path=tokenizer,
                                                            batch_size=10,
                                                            # can set as None if you do not have the ground truth
                                                            example_outputs=task_outputs)
                            del model
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            # input("press any key to continue")
                            print(f"task{sub_dir},seed{seed},step{step},lora_num{lora_num},acc:{task_acc}")
                            task_perf_list.append(task_acc)
                        # break
                    avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
                    print("average perf:", avg_perf, "best perf:", max_perf)
                    result[(step,lora_num,lr_n)]["lorahub avg acc"][sub_dir]=avg_perf
                    result[(step,lora_num,lr_n)]["lorahub max acc"][sub_dir]=max_perf
                    save_name=f"epo{step}_train{example_num}_lora_num{lora_num}_lr{lr}_dorahub_l2.csv"
                    tmp_result=pd.DataFrame(result[(step,lora_num,lr_n)])
                    tmp_result.to_csv(os.path.join("results", save_name))
                    step_result[step]=(avg_perf,max_perf)
                
            # print("step_result:",step_result)
            # break
        # except Exception as e:
        #     print("error:",e)
        #     continue
    result_pd=pd.DataFrame(result)
    return result,result_pd
if __name__ == "__main__":
    result_folder = "results"
    # zero_result,zero_result_df=evaluate_flan_results_zero_shot("data_bbh", "google/flan-t5-large")
    # zero_result_df.to_csv(os.path.join(result_folder, "zero_result.csv"))
    # # five shot for flan models
    # few_result,few_result_df=evaluate_flan_results_few_shot("data_bbh", "google/flan-t5-large")
    # few_result_df.to_csv(os.path.join(result_folder, "few_result.csv"))
    # five shot for lorahub models
    lorahub_result,lorahub_result_df=evaluate_lorahub_results_few_shot("data_bbh", "google/flan-t5-large")
    lorahub_result_df.to_csv(os.path.join(result_folder, "lorahub_result_dora.csv"))