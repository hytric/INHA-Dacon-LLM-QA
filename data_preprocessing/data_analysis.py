# data_analysis.py: 데이터 분석 및 전처리를 위한 코드
import pandas as pd
from transformers import AutoTokenizer
import re
from collections import Counter
from tqdm import tqdm

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('x2bee/POLAR-14B-v0.5')

# CSV 파일 로드
def load_csv(filepath):
    return pd.read_csv(filepath)

train_data = load_csv('./dataset/train.csv')

# 정답의 단어 수 (띄어쓰기 기준) 구하는 함수
def count_words(text):
    return len(text.split())

# 각 항목 처리 및 정답의 단어 수 구하기
num_entries = len(train_data)
word_counts = []
selected_ids = []

for index, entry in tqdm(train_data.iterrows(), total=num_entries, desc="Processing"):
    context_tokens = tokenizer.encode(entry['context'], add_special_tokens=False)
    question_tokens = tokenizer.encode(entry['question'], add_special_tokens=False)
    answer_tokens = tokenizer.encode(entry.get('answer', '').strip(), add_special_tokens=False)  # answer가 없을 경우 빈 문자열 처리

    answer = entry.get('answer', '').strip()
    word_count = count_words(answer)
    word_counts.append(word_count)
    
    if 16 <= word_count <= 16:
        selected_ids.append(entry['id'])

# 단어 수별 빈도수 계산
word_count_freq = Counter(word_counts)

# 단어 수별 빈도수 출력
print("Word Count Frequency:")
for word_count, freq in sorted(word_count_freq.items()):
    print(f'Word Count: {word_count}, Frequency: {freq}')

# 단어 수가 10개에서 16개 사이인 정답들의 ID 출력
print("\nSelected IDs (10 to 16 words):")
for id_ in selected_ids:
    print(id_)



#------------------------------------------------------------
# 정답이 context에 몇 개 들어가 있는지 카운트
import pandas as pd
from tqdm import tqdm

# CSV 파일 로드
def load_csv(filepath):
    return pd.read_csv(filepath)

train_data = load_csv('./dataset/train.csv')

# 정답이 context에 몇 개 들어가 있는지 카운트
answer_counts = []

for index, entry in tqdm(train_data.iterrows(), total=len(train_data), desc="Processing"):
    context = entry['context']
    answer = entry.get('answer', '').strip()
    
    # answer가 context에 몇 번 포함되어 있는지 카운트
    count = context.count(answer)
    
    answer_counts.append(count)

# 데이터 프레임에 결과 추가
train_data['answer_count_in_context'] = answer_counts

# 개수별로 정리
count_summary = train_data['answer_count_in_context'].value_counts().sort_index().reset_index()
count_summary.columns = ['answer_count_in_context', 'frequency']

# answer_count_in_context가 22번 이상인 id를 저장
ids_with_high_count = train_data[train_data['answer_count_in_context'] >= 43]['id'].tolist()

# 결과 확인
print("Count Summary:")
print(count_summary)

print("\nIDs with answer_count_in_context >= 43:")
print(ids_with_high_count)



#------------------------------------------------------------
# 문장 분할 및 문장 개수별 분포 확인
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
import json
from collections import Counter

# NLTK 데이터 다운로드 (최초 실행 시 필요)
nltk.download('punkt')

# CSV 파일 로드
def load_csv(filepath):
    return pd.read_csv(filepath)

train_data = load_csv('./dataset/train.csv')

# 문장을 분할하는 함수
def split_sentences(text):
    return sent_tokenize(text)

# 각 context를 문장 단위로 분할하고 새로운 열에 저장
all_sentences = []
sentence_counts = []
for index, entry in tqdm(train_data.iterrows(), total=len(train_data), desc="Processing"):
    context = entry['context']
    sentences = split_sentences(context)
    all_sentences.append(sentences)
    sentence_counts.append(len(sentences))

# 원본 데이터프레임에 문장 분할 결과를 새로운 열로 추가
train_data['sentences'] = all_sentences
train_data['sentence_count'] = sentence_counts

# 결과를 JSON 파일로 저장
output_data = train_data.to_dict(orient='records')
with open('processed_data.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

# 문장 개수별로 정리하여 출력 및 각 문장 개수에 해당하는 context id 뽑기
sentence_count_distribution = Counter(sentence_counts)
print("Sentence Count Distribution:")
for count, num_contexts in sorted(sentence_count_distribution.items()):
    ids = train_data[train_data['sentence_count'] == count].index.tolist()
    print(f"{count} sentences: {num_contexts} contexts, IDs: {ids}")

# 결과 확인
print(train_data[['context', 'sentence_count']].head())

#------------------------------------------------------------
# context 길이 분포 확인
import pandas as pd
from transformers import AutoTokenizer
import json
from tqdm import tqdm

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained('x2bee/POLAR-14B-v0.5')

# CSV 파일 로드
def load_csv(filepath):
    return pd.read_csv(filepath)

# 데이터 로드
train_data = load_csv('./dataset/train.csv')

# 데이터 정리
data_entries = []

for _, entry in tqdm(train_data.iterrows(), total=len(train_data), desc="Processing"):
    context_tokens = tokenizer.encode(entry['context'], add_special_tokens=False)
    question_tokens = tokenizer.encode(entry['question'], add_special_tokens=False)
    answer_tokens = tokenizer.encode(entry.get('answer', '').strip(), add_special_tokens=False)
    
    num_tokens = len(context_tokens) + len(question_tokens) + len(answer_tokens)
    
    data_entries.append({
        'id': entry['id'],
        'context': entry['context'],
        'question': entry['question'],
        'answer': entry.get('answer', '').strip(),
        'num_tokens': num_tokens
    })

# 토큰 수에 따라 내림차순으로 정렬
sorted_data_entries = sorted(data_entries, key=lambda x: x['num_tokens'], reverse=True)

# JSON 파일로 저장
with open('sorted_train_data.json', 'w', encoding='utf-8') as f:
    json.dump(sorted_data_entries, f, ensure_ascii=False, indent=4)

print("Data has been sorted and saved to sorted_train_data.json")