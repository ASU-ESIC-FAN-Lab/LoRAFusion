import os
import json
from LoRAs.baseLearner import myBaseLearner
from LoRAs.loraLearner import loraLeaner
from LoRAs.doraLeaner import doraLeaner
from LoRAs.veraLearner import veraLearner
from LoRAs.loraFusionLearner import loraFusionLearner
from LoRAs.constant import LORA_MODULE_NAMES
from args_helper import parse_args
import random
import pandas as pd
import torch
import wandb

def separate_valid_dataset(example_inputs, examples_outputs, valid_ratio=0.1):
    example_num = len(example_inputs)
    valid_num = int(example_num * valid_ratio)
    valid_inputs, valid_outputs = example_inputs[:valid_num], examples_outputs[:valid_num]
    train_inputs, train_outputs = example_inputs[valid_num:], examples_outputs[valid_num:]
    return train_inputs, train_outputs, valid_inputs, valid_outputs
def evaluate_lorahub_results_zero_shot(data_folder,args):
    log_experiment = args.log
    epoch = args.epoch
    lora_num = args.lora_num
    lr = args.lr
    load_in_4bit = args.load_in_4bit
    sub_dirs = os.listdir(data_folder)
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
            example_file_path = os.path.join(data_folder, sub_dir, "example.jsonl")
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
            # example_inputs, examples_outputs = example_inputs[:example_num], examples_outputs[:example_num]
            # separate the training and validation dataset
            train_inputs, train_outputs, valid_inputs, valid_outputs = separate_valid_dataset(example_inputs[:example_num], examples_outputs[:example_num], valid_ratio=0.05)
            # load the zero-shot examples for evaluation
            test_file_path = os.path.join(data_folder, sub_dir, "zero_shot.jsonl")
            task_inputs, task_outputs = [], []
            for line in open(test_file_path, "r", encoding="utf-8"):
                example = json.loads(line)
                task_inputs.append(example["context"])
                task_outputs.append(example["completion"])
            all_inputs= example_inputs+tuple(task_inputs)
            all_outputs=examples_outputs+tuple(task_outputs)
            # print(len(all_inputs))
            step_result={}
            
            task_perf_list = []
            
            if (epoch,lora_num,lr) not in result.keys():
                result[(epoch,lora_num,lr)]={'lorahub avg acc':{},'lorahub max acc':{}}

            for seed in range(1,4):
                if log_experiment:
                    wandb_config={"epochs":epoch,"lr":lr ,"task_name":sub_dir,"seed":seed}
                    wandb.init(project="dorahub_dora4bit",name=f"{sub_dir}",config=wandb_config)
                # lr=0.001
                random.seed(seed)
                print("Evaluating on task (lorahub): ", sub_dir, "with seed:", seed)

                def get_lora_module_list(lora_num=40):
                    return random.sample(LORA_MODULE_NAMES, lora_num) #what 
                # get a list of modules to be used in the composition
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated('cuda')} bytes")
                model = veraLearner(train_input=train_inputs,
                                        train_output=train_outputs,
                                        max_step=epoch,
                                        batch_size=args.batch_size,
                                        lr=lr,
                                        valid_input=valid_inputs,
                                        valid_output=valid_outputs,
                                        log_experiment=log_experiment,
                                        early_stopping=False,
                                        load_in_4bit=load_in_4bit)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                model.train()
                _, task_acc=model.inference(example_inputs=task_inputs, example_outputs=task_outputs)
                # _, task_acc=model.inference(example_inputs=all_inputs, example_outputs=all_outputs)
                class_name=model.__class__.__name__
                del model
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated('cuda')} bytes")
                torch.cuda.empty_cache()
                # input("press any key to continue")
                torch.cuda.reset_peak_memory_stats()
                # input("press any key to continue")
                print(f"task{sub_dir},seed{seed},epoch{epoch},lora_num{lora_num},acc:{task_acc}")
                if log_experiment:
                    wandb.log({"task_acc":task_acc})
                    wandb.finish()
                task_perf_list.append(task_acc)
            # break
            avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
            print("average perf:", avg_perf, "best perf:", max_perf)
            result[(epoch,lora_num,lr)]["lorahub avg acc"][sub_dir]=avg_perf
            result[(epoch,lora_num,lr)]["lorahub max acc"][sub_dir]=max_perf
            save_name=f"epo{epoch}_train{example_num}_lora_num{lora_num}_lr{lr}_f{class_name}_dt4bit.csv"
            tmp_result=pd.DataFrame(result[(epoch,lora_num,lr)])
            tmp_result.to_csv(os.path.join("results", save_name))
            step_result[epoch]=(avg_perf,max_perf)
                
    result_pd=pd.DataFrame(result)
    print("end of training---------------------------------")
    return result,result_pd
if __name__ == "__main__":
    if not os.path.exists("data_bbh"):
        # download dataset
        os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        # unzip
        os.system("unzip data_bbh.zip")
    result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    args = parse_args()
    # print(args)
    lorahub_result,lorahub_result_df=evaluate_lorahub_results_zero_shot("data_bbh",args)
    lorahub_result_df.to_csv(os.path.join(result_folder, "result.csv"))