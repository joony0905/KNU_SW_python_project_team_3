import pandas as pd
from utils.file_io import load_csv  
import os

def load_and_prepare_data(
    phishing_path='data/raw/init_phishing.csv',  # 기본 경로 설정
    normal_path='data/raw/generated_30k_ad_messages.csv',
    chat_path='data/raw/comments_1.csv',
    phishing_sample_size=7500,
    chat_sample_size=8000
):
    try:
        phishing_df = load_csv(phishing_path, usecols=["Spam message"], encoding='cp949')
        normal_df = load_csv(normal_path, encoding='cp949')
        chat_df = load_csv(chat_path, encoding='cp949')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"데이터 파일 로드 중 오류 발생: {e}")

    chat_df = chat_df.sample(n=chat_sample_size, random_state=42)
    phishing_df = phishing_df.sample(n=phishing_sample_size, random_state=42)

    normal_df['message'] = normal_df['Message']
    chat_df['message'] = chat_df['comment message']
    normal_df = pd.concat([normal_df, chat_df], ignore_index=True) #데이터 병합 과정
    normal_df = normal_df[['message']]

    phishing_df['message'] = phishing_df['Spam message']

    phishing_df['label'] = 1
    normal_df['label'] = 0

    combined_df = pd.concat([phishing_df, normal_df], ignore_index=True)
    combined_df = combined_df[['message', 'label']]

    return combined_df