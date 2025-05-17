import re
from konlpy.tag import Okt
import os

okt = Okt()

def clean_text(text):
    text = re.sub(r'ifg@|\*+|[^가-힣\s]', '', text)
    return text.strip()

def load_stopwords(filepath='data/stopwords.txt'):  # 기본 경로 설정
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            stopwords = [line.strip() for line in f]
        return stopwords
    except FileNotFoundError:
        raise FileNotFoundError(f"불용어 파일 '{filepath}'을 찾을 수 없습니다.")

stopwords = load_stopwords()

def tokenize_and_filter(text):
    tokens = okt.morphs(text, stem=True)
    filtered = [word for word in tokens if word not in stopwords and len(word) > 1]
    if not filtered:
        return " "  # 또는 다른 대체 문자열
    return ' '.join(filtered)

def preprocess_user_input(user_input):
    cleaned = clean_text(user_input)
    tokenized = tokenize_and_filter(cleaned)
    return tokenized