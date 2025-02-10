# QLoRA_Polar.py

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import json
import random
import transformers
import torch
import re
import string
import numpy as np
import os
import logging
from datetime import datetime

from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, StoppingCriteria, StoppingCriteriaList
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig
from datasets import Dataset, load_dataset, load_metric
from trl import SFTTrainer
from transformers import EvalPrediction

import wandb

#-------------------------------------------------------------------------
# prepare

os.environ['WANDB_API_KEY'] = ''
os.environ["WANDB_PROJECT"] = "POLAR-14B-v0.5"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

# 로그 디렉터리 생성
log_dir = "./log"
os.makedirs(log_dir, exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(os.path.join(log_dir, "training.log"), mode='w')])

import torch

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    transformers.set_seed(seed)  # for Hugging Face transformers
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logging.info(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logging.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logging.warning("No GPU found. A GPU is needed for quantization.")

# gpu 동작 여부 체크
check_gpu()

# 모든 랜덤성 제어
set_seed(42)

#-------------------------------------------------------------------------
# model setting

model_id = "x2bee/POLAR-14B-v0.5"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

logging.info("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    eos_token='</s>'
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    rope_scaling={"type": "dynamic", "factor": 2}
)

model.config.use_cache = False
model.config.pretraining_tp = 1

logging.info("Model and tokenizer loaded.")

#-------------------------------------------------------------------------
# dataset

def trans(x):
    return {
        'text': f"본문을 읽고 질문에 올바른 답변하시오. 본문: {x['context']} ### 질문: {x['question']} ###  답: {x['answer']}",
    }

def trans_eval(x):
    return {
        'text': f"본문을 읽고 질문에 올바른 답변하시오. 본문: {x['context']} 질문: {x['question']} ### 답: ",
    }

def trans_true(x):
    return {
        'text': f"본문을 읽고 질문에 올바른 답변하시오. 본문: {x['context']} ### 질문: {x['question']} ###  답: {x['answer']}",
    }


# 데이터 불러오기 및 변환
logging.info("Loading and transforming datasets...")
train_data = Dataset.from_json('train_data.json')
eval_data = Dataset.from_json('eval_data.json')
true_data = Dataset.from_json('eval_data.json')

# 변환 함수를 적용합니다.
train_data = train_data.map(trans)
eval_data = eval_data.map(trans_eval)
true_data = true_data.map(trans_true)

logging.info("Datasets loaded and transformed.")

#-------------------------------------------------------------------------
# training

model.train()
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

output_path = "./result/POLAR-14B-v0.5/"

config = LoraConfig(
    lora_alpha=256,
    lora_dropout=0.05,
    r=128,
    target_modules=["q_proj","up_proj","o_proj","k_proj","down_proj","gate_proj","v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

train_params = TrainingArguments(
    output_dir=output_path,
    run_name="POLAR-14B-v0.5-V2",  # 여기에 원하는 run name을 설정
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
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
    per_device_eval_batch_size=2,
    eval_accumulation_steps=2,
    evaluation_strategy="epoch"
)

def compute_metrics(pred: EvalPrediction):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids

    # 예측값과 실제값을 텍스트로 디코딩하기 전에 범위 검증 및 클리핑
    vocab_size = tokenizer.vocab_size
    labels = np.clip(labels, 0, vocab_size - 1)

    # 예측값과 실제값을 텍스트로 디코딩
    preds_text = [tokenizer.decode(pred_seq, skip_special_tokens=True) for pred_seq in preds]
    labels_text = [tokenizer.decode(label_seq.tolist(), skip_special_tokens=True) for label_seq in labels]

    # 평가 결과를 저장할 리스트
    result = []

    # 예측 텍스트와 실제 텍스트를 비교하여 F1 점수 계산
    f1_scores = []
    for idx, pred_text in enumerate(preds_text):
        true_text = true_data[idx]
        
        pred_tokens = pred_text.split()
        true_tokens = true_text['text']
        
        # 공통 토큰의 수 계산
        common_tokens = set(pred_tokens) & set(true_tokens)
        num_common = len(common_tokens)
        
        # Precision, Recall, F1 계산
        if len(pred_tokens) > 0 and len(true_tokens) > 0:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(true_tokens)
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
        else:
            f1_score = 0.0

        f1_scores.append(f1_score)

        evaluation_result = {
            "prediction": pred_text,
            "answer": true_text,
            "f1_score": f1_score
        }
        result.append(evaluation_result)

    # 전체 평균 F1 점수 계산
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # 현재 시간으로 파일 이름 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = output_path + f'evaluation_results_{current_time}.json'
    
    # 모든 평가 결과를 하나의 JSON 파일로 저장
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    return {'f1': average_f1, 'result_file': file_name}


logging.info("Initializing trainer...")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=config,
    dataset_text_field='text',
    tokenizer=tokenizer,
    args=train_params,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics
)

logging.info("Starting training...")

trainer.train()

#-------------------------------------------------------------------------
# saving

# 모델과 토크나이저 저장
logging.info("Saving model and tokenizer...")
trainer.model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# 추가적인 config 저장
model.config.save_pretrained(output_path)
config.save_pretrained(output_path)

logging.info("Model, tokenizer, and config have been saved.")
