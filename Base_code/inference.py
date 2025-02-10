# inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
import pandas as pd


# 모델과 토크나이저 불러오기
output_path = "./result/polyglot-ko-12/"  # 실제 모델 경로로 변경

tokenizer = AutoTokenizer.from_pretrained(output_path)
model = AutoModelForCausalLM.from_pretrained(output_path)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

# 테스트 데이터 로드 및 추론 수행
file_path = './dataset/test.csv'
test_data = pd.read_csv(file_path)
submission_dict = {}

for index, row in test_data.iterrows():
    try:
        context = row['context']
        question = row['question']
        id = row['id']

        if context is not None and question is not None:
            question_prompt = f"당신은 본문을 읽고 질문에 답변하는 역할을 하는 챗봇입니다. 사용자의 질문에 올바른 답변을 하세요.\n###본문: {x['context']} 질문: {x['question']}\n### 답변: <|endoftext|>"

            answer = generate_response(question_prompt)
            submission_dict[id] = answer
        else:
            submission_dict[id] = 'Invalid question or context'

    except Exception as e:
        print(f"Error processing question {id}: {e}")
        submission_dict[id] = 'Error processing question'

# 결과 저장
df = pd.DataFrame(list(submission_dict.items()), columns=['id', 'answer'])
df.to_csv('./submission.csv', index=False)
