# similarity_DP.py

import json
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import nltk
from konlpy.tag import Okt
import numpy as np
import re
import pandas as pd

# Ensure you have the necessary NLTK data files
nltk.download('punkt')

# Manually defined list of Korean stop words
korean_stop_words = set([
    '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '에서', '부터', '까지', '으로', '로', '보다', '와', '고', '도', '나', 
    '마저', '그리고', '그러나', '그래서', '그렇지만', '그런데', '따라서', '또는', '혹은', '저', '나', '너', '그', '그녀', '우리', '당신', 
    '여러분', '이것', '저것', '그것', '아', '오', '어', '야', '와', '허', '매우', '너무', '아주', '좀', '많이', '잘', '덜', '꼭', '대체로', 
    '역시', '특히', '다시', '마치', '또', '그냥', '즉', '거의', 
])

def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub(r"'", " ", text)
        text = re.sub(r'"', " ", text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\(사진\)', ' ', text)
        text = re.sub(r'△', ' ', text)
        text = re.sub(r'▲', ' ', text)
        text = re.sub(r'◇', ' ', text)
        text = re.sub(r'■', ' ', text)
        text = re.sub(r'ㆍ', ' ', text)
        text = re.sub(r'↑', ' ', text)
        text = re.sub(r'·', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.sub(r'=', ' ', text)
        text = re.sub(r'사례', ' ', text)
        return text

    def remove_parentheses_text(text):
        '''괄호 안의 띄어쓰기 및 한자만 포함된 텍스트 제거'''
        def should_remove(match):
            content = match.group(1).strip()
            # 띄어쓰기와 한자만 있는 경우 제거
            return re.fullmatch(r'[\s\u4E00-\u9FFF]*', content) is not None

        return re.sub(r'\(([^)]*)\)', lambda x: '' if should_remove(x) else x.group(0), text)
        
    def white_space_fix(text):
        '''연속된 공백일 경우 하나의 공백으로 대체'''
        return ' '.join(text.split())

    # 전처리 함수들을 차례로 적용
    return white_space_fix(remove_parentheses_text(remove_(s)))

def jaccard_similarity(query, document):
    query_tokens = set(query)
    document_tokens = set(document)
    intersection = query_tokens.intersection(document_tokens)
    union = query_tokens.union(document_tokens)
    return len(intersection) / len(union)

def tokenize_korean(text, stop_words, method='pos'):
    okt = Okt()
    if method == 'morphs':
        tokens = [word for word in okt.morphs(text) if word not in stop_words]
    elif method == 'pos':
        tokens = [word for word, pos in okt.pos(text) if word not in stop_words]
    elif method == 'nouns':
        tokens = [word for word in okt.nouns(text) if word not in stop_words]
    return tokens

def process_json(data):
    # Check if a GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the SBERT model and move it to the appropriate device
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    model = model.to(device)

    output_data = []

    # Process each entry in the input JSON
    for entry in tqdm(data, desc="Processing entries"):
        ID = entry['id']
        context = entry['context']
        question = entry['question']
        answer = entry['answer']

        # Split the context into sentences using NLTK
        sentences = nltk.sent_tokenize(context)

        # Encode the question and sentences
        question_embedding = model.encode(question, convert_to_tensor=True, device=device)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True, device=device)

        # Compute cosine similarity between the question and each sentence
        cosine_similarities = util.pytorch_cos_sim(question_embedding, sentence_embeddings).squeeze().cpu().numpy()

        # Tokenize the question with stop words removal
        question_tokens = tokenize_korean(question, korean_stop_words, method='pos')

        # Compute Jaccard similarity between the question and each sentence
        jaccard_similarities = []
        for sentence in sentences:
            sentence_tokens = tokenize_korean(sentence, korean_stop_words, method='pos')
            similarity = jaccard_similarity(question_tokens, sentence_tokens)
            jaccard_similarities.append(similarity)

        jaccard_similarities = np.array(jaccard_similarities)

        # Calculate total similarities
        total_similarities = cosine_similarities + 4 * jaccard_similarities

        # Get the indices of the top 10 sentences based on total similarity
        num_sentences = min(10, len(sentences))
        top_indices = total_similarities.argsort()[-num_sentences:][::-1]

        # Filter context to keep only the selected top 10 sentences, ensuring no duplicates
        filtered_context = ' '.join(dict.fromkeys([sentences[i] for i in sorted(top_indices)]))

        output_data.append({
            "id": ID,
            "context": filtered_context,
            "question": question,
            "answer": answer,
        })

    return output_data

# CSV 파일 경로
csv_file = './dataset/train.csv' #test.csv
data = pd.read_csv(csv_file)

json_data = []

# 데이터 전처리
for _, row in tqdm(data.iterrows(), total=data.shape[0]):
    ID = row['id']
    context = normalize_answer(row['context'])  # context 열에 전처리 함수 적용
    question = normalize_answer(row['question'])
    answer = row['answer']

    json_data.append({
        "id": ID,
        "context": context,
        "question": question,
        "answer": answer,
    })

# 유사도 계산 및 문장 필터링
filtered_data = process_json(json_data)

# JSON 형식으로 저장
json_string = json.dumps(filtered_data, ensure_ascii=False, indent=4)
with open('train_preprocessed4.json', 'w', encoding='utf-8') as file:
    file.write(json_string)

print("Done!")
