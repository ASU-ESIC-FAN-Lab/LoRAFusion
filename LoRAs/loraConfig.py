from peft import LoraConfig, PeftType
lora_config = LoraConfig(
    peft_type=PeftType.LORA,
    base_model_name_or_path="google/flan-t5-large",
    task_type="SEQ_2_SEQ_LM",
    inference_mode=True,
    r=16,
    target_modules={"v", "q"},
    lora_alpha=32,
    lora_dropout=0.1
)