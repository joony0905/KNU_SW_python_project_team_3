import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import swifter
import joblib
from data_loader import load_and_prepare_data
from utils.file_io import save_dataframe_to_csv, load_csv
from utils.preprocessing import clean_text, tokenize_and_filter
import os

swifter.set_defaults(display_progress=True)
swifter.set_defaults(allow_dask_on_strings=False)

def train_and_evaluate_model(
    combined_df_path='data/processed/cleaned_data.csv',
    vectorizer_path='model/vectorizer.pkl',
    model_path='model/spam_classifier.pkl',
    raw_phishing_path='data/raw/init_phishing.csv',
    raw_normal_path='data/raw/generated_30k_ad_messages.csv',
    raw_chat_path='data/raw/comments_1.csv',
    test_size=0.2,
    random_state=42,
    max_features=15000
):
    try:
        if os.path.exists(combined_df_path):
            print(f"{combined_df_path} 파일이 존재합니다. 전처리 과정을 건너뛰고 바로 로드합니다.")
            combined_df = load_csv(combined_df_path, encoding = 'utf-8-sig')
        else:
            print(f"{combined_df_path} 파일이 존재하지 않습니다. 데이터 로딩 및 전처리를 시작합니다.")
            combined_df = load_and_prepare_data(
                phishing_path=raw_phishing_path,
                normal_path=raw_normal_path,
                chat_path=raw_chat_path
            )

            combined_df['message'] = (
                combined_df['message']
                .astype(str)
                .swifter.apply(lambda x: tokenize_and_filter(clean_text(x)))
            )
            save_dataframe_to_csv(combined_df, combined_df_path)
    except FileNotFoundError as e:
        print(f"오류: {e}")
        return  

    X = combined_df['message']
    y = combined_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    # 벡터라이저 로드 또는 생성
    if os.path.exists(vectorizer_path):
        while True:
            try:
                flag = int(input("벡터라이저가 이미 존재합니다. 새로 생성하시려면 0 / 기존 파일을 Load하려면 1을 입력하세요. "))
                if(flag == 1):           
                    vectorizer = joblib.load(vectorizer_path)
                    print(f"벡터라이저 '{vectorizer_path}'를 로드했습니다.")
                    break
                elif(flag == 0):
                    vectorizer = TfidfVectorizer(max_features=max_features)
                    print("벡터라이저를 새로 생성합니다.")
                    break
                else:
                    print("잘못된 값입니다. 다시 입력해주세요.")
            except:
                print("잘못된 값입니다. 다시 입력해주세요.")
    else:
        vectorizer = TfidfVectorizer(max_features=max_features)
        print(f"벡터라이저 '{vectorizer_path}'가 없어 새로 생성합니다.")

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 모델 로드 또는 생성
    if os.path.exists(model_path):
        while True:
            try:
                flag = int(input("기존 학습 모델이 이미 존재합니다. 새로 생성하시려면 0 / 기존 파일을 Load하려면 1을 입력하세요. "))
                if(flag == 1):           
                    model = joblib.load(model_path)
                    print(f"기존 학습 모델 '{model_path}'를 로드했습니다.")
                    break
                elif(flag == 0):
                    print("모델을 새로 생성하여 학습합니다.")
                    model = MultinomialNB()
                    model.fit(X_train_vec, y_train)
                    break
                else:
                    print("잘못된 값입니다. 다시 입력해주세요.")
            except:
                print("잘못된 값입니다. 다시 입력해주세요.")
    else:
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        print(f"모델 '{model_path}'가 없어 새로 학습합니다.")

    y_pred = model.predict(X_test_vec)

    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    flag = int(input("현재 벡터라이저와 학습 모델을 저장하시겠습니까? 1: 예 / 0: 아니오 " ))
    while True:
        try:
            if(flag == 1):
                joblib.dump(model, model_path) 
                joblib.dump(vectorizer, vectorizer_path)
                print("벡터라이저와 모델을 저장했습니다.")
                break
            elif(flag == 0):
                return
            else:
                print("잘못된 값입니다. 다시 입력해주세요.")
        except:
            print("잘못된 값입니다. 다시 입력해주세요.")

if __name__ == '__main__':
    train_and_evaluate_model()