import pandas as pd
import os

def load_csv(filepath='data/processed/cleaned_data.csv', **kwargs):  # 기본 경로 설정
    try:
        return pd.read_csv(filepath, **kwargs)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV 파일 '{filepath}'을 찾을 수 없습니다.")

def save_dataframe_to_csv(df, filepath='data/processed/cleaned_data.csv', index=False, encoding='utf-8-sig'):  # 기본 경로 설정
    df.to_csv(filepath, index=index, encoding=encoding)