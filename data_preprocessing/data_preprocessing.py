# data_preprocessing.py

import pandas as pd
import re
import json

# Normalization function
def normalize_answer(s):
    def remove_(text):
        ''' 불필요한 기호 제거 '''
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

    def remove_hanja(text):
        '''한자 제거'''
        return re.sub(r'[\u4E00-\u9FFF]+', '', text)

    def white_space_fix(text):
        '''연속된 공백일 경우 하나의 공백으로 대체'''
        return ' '.join(text.split())

    return white_space_fix(remove_hanja(remove_(s)))

# Load the CSV file
df = pd.read_csv('./dataset/train.csv')

# Add " at the end of each answer
df['answer'] = df['answer'] + '<|end_of_text|>'

# Normalize context, question, and answer
df['context'] = df['context'].apply(normalize_answer)
df['question'] = df['question'].apply(normalize_answer)
df['answer'] = df['answer'].apply(normalize_answer)

# Convert the DataFrame to a dictionary
data = df.to_dict(orient='records')

# Save to JSON file
with open('train_base.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Preprocessing complete. The file has been saved.")