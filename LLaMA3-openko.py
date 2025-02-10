# LLaMA3-openko.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json
import torch
import logging
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTTrainer
from accelerate import Accelerator
from collections import Counter

MODEL_ID = "beomi/Llama-3-Open-Ko-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

logging.info("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    eos_token='<|end_of_text|>'
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # rope_scaling={"type": "dynamic", "factor": 2}
)

model.config.use_cache = False
model.config.pretraining_tp = 1

def trans(x):
    return {'text': f"질문에 대한 답변을 맥락에서 찾은 후 최대한 간략하게 대답해줘.\n\n### 질문: {x['question']}\n\n### 맥락: {x['context']}\n\n### 답변: {x['answer']}" }

data = Dataset.from_json('train_base.json')
train_data = data.map(lambda x: trans(x))

model.train()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

OUTPUT_PATH = "./results/llama3/"

config = LoraConfig(
    lora_alpha=16,  # zero-LoRA 설정
    lora_dropout=0.05,
    r=16,
    target_modules=["q_proj","up_proj","o_proj","k_proj","down_proj","gate_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

train_params = TrainingArguments(
    output_dir=OUTPUT_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    save_strategy="epoch", 
    optim="paged_adamw_8bit",
    learning_rate=1e-4,
    logging_steps=100,
    weight_decay=0.01,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    fp16=True,
    lr_scheduler_type="cosine",
    seed=42
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=config,
    dataset_text_field='text',
    tokenizer=tokenizer,
    args=train_params,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_PATH)