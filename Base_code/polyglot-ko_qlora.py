# polyglot-ko_qlora.py

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import json
import random
import transformers
import torch
import re
import string

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig
from datasets import Dataset, load_dataset, load_metric
from trl import SFTTrainer

import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    transformers.set_seed(seed)  # for Hugging Face transformers
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU found. A GPU is needed for quantization.")

# gpu 동작 여부 체크
check_gpu()

# 모든 랜덤성 제어
set_seed(42)

model_id = "EleutherAI/polyglot-ko-12.8b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    eos_token="<|endoftext|>"
    )

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    rope_scaling = {"type": "dynamic", "factor": 2}
)

model.config.use_cache=False
model.config.pretraining_tp=1


def trans(x):
    return {
        'text': f"당신은 본문을 읽고 질문에 답변하는 역할을 하는 챗봇입니다. 사용자의 질문에 올바른 답변을 하세요.\n###본문: {x['context']} 질문: {x['question']}\n### 답변: {x['answer']}<|endoftext|>"}


data = Dataset.from_json('train_json.json')
train_data = data.map(lambda x: trans(x))

model.train()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

output_path = "./result/polyglot-ko-12/"

config = LoraConfig(
    lora_alpha=256,
    lora_dropout=0.05,
    r=128,
    target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
    bias="none",
    task_type="CAUSAL_LM"
)

train_params = TrainingArguments(
    output_dir=output_path,
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
    seed=42,
    evaluation_strategy = "epoch",
    eval_steps = 100
)

metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=config,
    dataset_text_field='text',
    tokenizer=tokenizer,
    args=train_params,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics
)

trainer.train()

# 모델과 토크나이저 저장
trainer.model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# 추가적인 config 저장
model.config.save_pretrained(output_path)
config.save_pretrained(output_path)

print("Model, tokenizer, and config have been saved.")