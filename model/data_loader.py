import pandas as pd
import numpy as np
from utils.file_io import rf_load_csv, naive_load_csv  
import os

def naive_load_and_prepare_data(
    phishing_path='data/raw/init_phishing.csv',  # 기본 경로 설정
    normal_path='data/raw/generated_30k_ad_messages.csv',
    chat_path='data/raw/comments_1.csv',
    phishing_sample_size=7500,
    chat_sample_size=8000
):
    try:
        phishing_df = naive_load_csv(phishing_path, usecols=["Spam message"], encoding='cp949')
        normal_df = naive_load_csv(normal_path, encoding='cp949')
        chat_df = naive_load_csv(chat_path, encoding='cp949')
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

def rf_load_and_prepare_data(
    phishing_path='data/raw/init_phishing.csv',
    normal_files=None,
    phishing_sample_size=30000,
    encoding_phishing='cp949'
):
    if normal_files is None:
        normal_files = [
        ("data/raw/generated_30k_ad_messages.csv", "ad", 22000),
        ("data/raw/bank_messages.csv", "bank", 2000),
        ("data/raw/parcel_messages.csv", "parcel", 2000),
        ("data/raw/telecom_messages.csv", "telcom", 5000),
        ("data/raw/university_messages.csv", "university", 2000), 
        ("data/raw/comments_1.csv", "chat", 7000)
        ]

    def load_normal(file_path, category, n=None):
        try:
            df = rf_load_csv(file_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            print(f"{file_path} → utf-8-sig 인코딩 실패. cp949로 재시도합니다.")
            df = rf_load_csv(file_path, encoding='cp949')

        df = df.rename(columns={df.columns[0]: 'message'})
        if n:
            df = df.sample(n=n, random_state=42)
        df['label'] = 'ham'
        df['category'] = category
        return df[['message', 'label', 'category']]

    try:
        normal_dfs = [load_normal(path, cat, n) for path, cat, n in normal_files]
        normal_df = pd.concat(normal_dfs, ignore_index=True)

        spam_df = pd.read_csv(phishing_path, usecols=["Spam message"], encoding=encoding_phishing)
        spam_df = spam_df.rename(columns={"Spam message": "message"})
        spam_df['label'] = 'spam'
        spam_df['category'] = np.nan

        spam_df = spam_df.sample(n=phishing_sample_size, random_state=42)

        combined_df = pd.concat([normal_df, spam_df], ignore_index=True)

        return combined_df


    except FileNotFoundError as e:
        raise FileNotFoundError(f"데이터 파일 로드 중 오류 발생 data/raw를 확인해주세요.: {e}")

