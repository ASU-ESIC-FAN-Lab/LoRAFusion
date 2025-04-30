
from peft import PeftConfig,get_peft_model
import torch.optim as optim
from lorahub.constant import LORA_MODULE_NAMES
import wandb
from lorahub.baseLearner import myBaseLearner
import bitsandbytes as bnb

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
                    early_stopping=False,
                    load_in_4bit=False,
                    load_in_8bit=False,
                    log_experiment=False,
                    **kwargs):
        super().__init__(model_name_or_path=model_name_or_path,
                            batch_size=batch_size,
                            seed=seed,
                            lr=lr,
                            max_step=max_step,
                            train_input=train_input,
                            train_output=train_output,
                            valid_input=valid_input,
                            valid_output=valid_output,
                            early_stopping=early_stopping,
                            load_in_4bit=load_in_4bit,
                            load_in_8bit=load_in_8bit,
                            log_experiment=log_experiment,
                            **kwargs)
    
    def _load_model(self):
        base_model = super()._load_model(train_base=False)
        lora_module= LORA_MODULE_NAMES[0]
        lora_config = PeftConfig.from_pretrained(lora_module)
        dora_config=lora_config
        dora_config.use_dora=True
        #lord new lora model
        dora_model = get_peft_model(base_model,dora_config)
        # for name, param in base_model.named_parameters():
        #     if "lora" in name:
        #         param.requires_grad = True
        # lora_model.print_trainable_parameters()
        return dora_model
    
    def train(self,validation=True):
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is required")
        print("start training")
        validation = validation and self.valid_dataloader is not None

        params_direction=[]
        params_magnitude=[]
        for name, param in self.model.named_parameters():
            # if "lora" in name:
            if "lora_magnitude_vector" in name:
                param.requires_grad = True
                params_magnitude.append(param)
            elif "lora" in name:
                param.requires_grad = True
                params_direction.append(param)
        self.model.print_trainable_parameters()
        print(len(params_direction),len(params_magnitude))
        if self.quantization_config is not None:
            # optimizer = bnb.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
            #                        lr=self.lr, optim_bits=32, percentile_clipping=95,weight_decay=0.00001)
            optimizer_direction = bnb.optim.Adam(params_direction,lr=self.lr,weight_decay=0.0001, percentile_clipping=95)
            optimizer_magnitude = bnb.optim.Adam(params_magnitude, lr=0.005, percentile_clipping=95)
        else:
            optimizer_direction = optim.Adam(params_direction, lr=self.lr,weight_decay=0.001)
            optimizer_magnitude = optim.Adam(params_magnitude, lr=0.005,weight_decay=0.000)
        # optimizer = bnb.optim.Adam8bit(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        #                             , betas=(0.9, 0.995), optim_bits=32, percentile_clipping=5)

        for step in range(self.max_step):
            total_loss = 0

            for _,batch in enumerate(self.train_dataloader):
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(self.device)} bytes")
                optimizer_direction.zero_grad()
                optimizer_magnitude.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss/len(batch["input_ids"])
                # print(f"Memory allocated for batch: {torch.cuda.memory_allocated(self.device)} bytes")
                # input("press any key to continue")
                total_loss += loss.item()
                loss.backward()
                self.check_nan_in_gradients(self.model)
                optimizer_direction.step()
                optimizer_magnitude.step()
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


                 
