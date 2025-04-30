

from peft import get_peft_model,VeraConfig
from lorahub.baseLearner import myBaseLearner

class veraLearner(myBaseLearner):
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
        vera_config=VeraConfig(r=128)
        model=get_peft_model(base_model,vera_config)
        model.eval()
        for name, param in model.named_parameters():
            if 'vera_lambda' in name:
                param.requires_grad = True
        # lora_model.print_trainable_parameters()
        return model
        

                 
