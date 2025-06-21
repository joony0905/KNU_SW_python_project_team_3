import sys 
import os

if __name__ == '__main__':
    ROOT_DIR = os.getcwd()  #상위 폴더도 인식하기 위함 
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import swifter
import joblib
from utils.model_utils import *
from data_loader import *
from utils.file_io import *
from utils.preprocessing import clean_text, tokenize_and_filter
from scipy.sparse import hstack, csr_matrix 
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

swifter.set_defaults(display_progress=True)
swifter.set_defaults(allow_dask_on_strings=False)

def naive_train_and_evaluate_model(
    combined_df_path='data/processed/cleaned_data.csv',
    vectorizer_path='model/nv_tf_idf_vectorizer.pkl',
    model_path='model/nv_spam_classifier.pkl',
    iso_model_path='model/isotonic_nv_spam_classifier.pkl',
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
            combined_df = naive_load_csv(combined_df_path, encoding = 'utf-8-sig')
        else:
            print(f"{combined_df_path} 파일이 존재하지 않습니다. 데이터 로딩 및 전처리를 시작합니다.")
            combined_df = naive_load_and_prepare_data(
                phishing_path=raw_phishing_path,
                normal_path=raw_normal_path,
                chat_path=raw_chat_path
            )

            combined_df['message'] = (
                combined_df['message']
                .astype(str)
                .swifter.apply(lambda x: tokenize_and_filter(clean_text(x)))
            )
            naive_save_dataframe_to_csv(combined_df, combined_df_path)
            print("전처리된 데이터를 저장했습니다.")
    except FileNotFoundError as e:
        print(f"오류: {e}")
        return  

    X = combined_df['message']
    y = combined_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

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

    #sigmo_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=3)
    #sigmo_model.fit(X_train_vec, y_train)

    #iso_model = CalibratedClassifierCV(estimator=model, method='isotonic', cv=5)
    #iso_model.fit(X_train_vec, y_train)

    plot_calibration_curve(model, X_test_vec, y_test, model_name="Naive_Bayes", n_bins=20)
    #plot_calibration_curve(sigmo_model, X_test_vec, y_test, model_name="sigmo_Naive_Bayes", n_bins=20)
    #plot_calibration_curve(iso_model, X_test_vec, y_test, model_name="iso_Naive_Bayes", n_bins=20)

    flag = int(input("현재 벡터라이저와 학습 모델을 저장하시겠습니까? 1: 예 / 0: 아니오 " ))
    while True:
        try:
            if(flag == 1):
                joblib.dump(model, model_path) 
                #joblib.dump(iso_model, iso_model_path)
                joblib.dump(vectorizer, vectorizer_path)
                print("벡터라이저와 모델을 저장했습니다.")
                break
            elif(flag == 0):
                return
            else:
                print("잘못된 값입니다. 다시 입력해주세요.")
        except:
            print("잘못된 값입니다. 다시 입력해주세요.")

def rf_train_and_evaluate_model(
    combined_df_path='data/processed/rf_cleaned_data.csv',
    vectorizer_path='model/rf_count_vectorizer.pkl',
    model_path='model/rf_spam_classifier.pkl',
    iso_model_path='model/rf_isotonic_spam_classifier.pkl',
    category_columns_path='model/rf_category_columns.pkl',
    test_size=0.2,
    random_state=42,
    max_features=10000):
    try:
        if os.path.exists(combined_df_path):
            print(f"{combined_df_path} 파일이 존재합니다. 전처리 과정을 건너뜁니다.")
            combined_df = pd.read_csv(combined_df_path, encoding='utf-8-sig')
        else:
            print(f"{combined_df_path} 파일이 존재하지 않습니다. 데이터를 로드하고 전처리합니다.")
        
            combined_df = rf_load_and_prepare_data()

            # 1. 전처리 결과를 따로 저장
            combined_df['processed'] = (
                combined_df['message']
                .astype(str).swifter.apply(lambda x: tokenize_and_filter(clean_text(x))))

            # 2. 전처리된 값 기준으로 중복 제거
            combined_df = combined_df.drop_duplicates(subset='processed').reset_index(drop=True)

            combined_df['label'] = combined_df['label'].map({'ham': 0, 'spam': 1})
            rf_save_dataframe_to_csv(combined_df, combined_df_path, index=False, encoding='utf-8-sig')
            print("전처리된 데이터를 저장했습니다.")

    except FileNotFoundError as e:
        print(f"오류: {e}")
        return

    if os.path.exists(vectorizer_path):
        while True:
            try:
                flag = int(input("Count 벡터라이저가 이미 존재합니다. 새로 생성 0 / 불러오기 1: "))
                if flag == 1:
                    vectorizer = joblib.load(vectorizer_path)
                    print("벡터라이저를 로드했습니다.")
                    #print(type(vectorizer))
                    break
                elif flag == 0:
                    vectorizer = CountVectorizer(
                    max_features=20000,
                    ngram_range=(1, 1), 
                    min_df=3,           
                    max_df=0.95           
                    )
                    print("새 벡터라이저를 생성합니다.")
                    break
                else:
                    print("0 또는 1을 입력해주세요.")
            except:
                print("잘못된 입력입니다.")
    else:
        vectorizer = CountVectorizer(
                    max_features=20000,
                    ngram_range=(1,1), 
                    min_df=3,            
                    max_df=0.95           
                    )
        print("벡터라이저가 없어 새로 생성합니다.")

    # 카테고리 원핫 전체 열 확보 (전체 컬럼 유지용)
    category_onehot = pd.get_dummies(combined_df['category'])
    
    #ad는 광고문자 총망라. 카테고리 제외
    category_onehot = category_onehot.drop(columns=['ad'], errors='ignore')
    
    # 3. 학습 시에도 'processed' 사용
    train_df, test_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df['label'], random_state=42)
    X_train = vectorizer.fit_transform(train_df['processed'])
    X_test = vectorizer.transform(test_df['processed']) 

    #문장 길이 feature 추가
    len_train = csr_matrix([[len(msg.split())] for msg in train_df['processed']])
    len_test = csr_matrix([[len(msg.split())] for msg in test_df['processed']])

    #one hot 없이 했을때랑 비교
    #X_train = vectorizer.fit_transform(train_df['processed'])  
    #X_test = vectorizer.transform(test_df['processed']) 

    # 카테고리 원핫 인코딩: 컬럼 일치 보장
    cat_train = pd.get_dummies(train_df['category']).reindex(columns=category_onehot.columns, fill_value=0)
    cat_test = pd.get_dummies(test_df['category']).reindex(columns=category_onehot.columns, fill_value=0)



    cat_train_sparse = csr_matrix(cat_train.values)
    cat_test_sparse = csr_matrix(cat_test.values)
    
    # 최종 결합
    X_train = hstack([X_train, len_train, cat_train_sparse])
    X_test = hstack([X_test, len_test, cat_test_sparse])
    y_train = train_df['label']
    y_test = test_df['label']

    if os.path.exists(model_path):
        while True:
            try:
                flag = int(input("모델이 이미 존재합니다. 새로 학습 0 / 불러오기 1: "))
                if flag == 1:
                    model = joblib.load(model_path)
                    print("모델을 로드했습니다.")
                    break
                elif flag == 0:
                    model = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=15,
                        min_samples_leaf=30,
                        min_samples_split=30,
                        max_features='sqrt',
                        class_weight=None,
                        random_state=42,
                        n_jobs=-1)
                    model.fit(X_train, y_train)
                    print("모델을 새로 학습했습니다.")
                    break
                else:
                    print("0 또는 1을 입력해주세요.")
            except:
                print("잘못된 입력입니다.")
    else:
        model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=15,
                    min_samples_leaf=30,
                    min_samples_split=30,
                    max_features='sqrt',
                    class_weight=None,
                    random_state=42,
                    n_jobs=-1)
        model.fit(X_train, y_train)
        print("모델이 없어 새로 학습했습니다.")

    y_pred = model.predict(X_test)
    print("정확도:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    #disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
    #disp.plot(cmap='Blues')
    #plt.title('Random Forest Confusion Matrix')
    #plt.show()

    # 기존 모델을 감싸서 보정
    # iso_model = CalibratedClassifierCV(estimator=model, method='isotonic', cv=5)
    # #sigmo_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)

    # iso_model.fit(X_train, y_train)
    # #sigmo_model.fit(X_train, y_train)
    # iso_y_pred = iso_model.predict(X_test)
    # print("Calibrated isotonic 보정 모델 \n정확도:", accuracy_score(y_test, iso_y_pred))
    # print(classification_report(y_test, iso_y_pred))

    #sigmo_model = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=5)
    #sigmo_model.fit(X_train, y_train)
    #plot_calibration_curve(model, X_test, y_test, model_name="general_RF_Model") #calib곡선 시각화
    #plot_calibration_curve(iso_model, X_test, y_test, model_name="iso_RF_Model")
    #plot_calibration_curve(sigmo_model, X_test, y_test, model_name="sigmo_RF_Model")

    while True:
        try:
            flag = int(input("학습 모델 파일들을 저장할까요? 1: 예 / 0: 아니오 "))
            if flag == 1:
                joblib.dump(model, model_path)
                joblib.dump(vectorizer, vectorizer_path)
                joblib.dump(category_onehot.columns.tolist(), category_columns_path)
                print("학습 모델 파일들들을 저장했습니다.")
                break
            elif flag == 0:
                break
            else:
                print("0 또는 1을 입력해주세요.")
        except:
            print("잘못된 입력입니다.")


def k_means_clustering(
    if_phishing_path='data/processed/if_cleand_data.csv',
    init_phishing_path = 'data/raw/init_phishing.csv',
    vectorizer_path='model/if_count_vectorizer.pkl',
    kmeans_model_path='model/kmeans_spam.pkl',
    num_clusters=2
):

    # 스팸 원본 데이터 로드
    if os.path.exists(if_phishing_path):
        spam_df = pd.read_csv(if_phishing_path, encoding='utf-8-sig')
        print(f"스팸 전처리 데이터 로드 완료: {len(spam_df)} rows")
    else:
        print(f"{if_phishing_path} 가 존재하지 않습니다. 전처리를 진행합니다.")
        spam_df = pd.read_csv(init_phishing_path, encoding='cp949')
        spam_df['message'] = (
                spam_df['Spam message']
                .astype(str)
                .swifter.apply(lambda x: tokenize_and_filter(clean_text(x)))
            )
        spam_df.to_csv(if_phishing_path, index=False, columns=['message'], encoding='utf-8-sig')
        print("전처리 데이터 저장 완료")

    # 벡터라이저 로드
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
        print("CountVectorizer 로드 완료")
    else:
        print(f"{vectorizer_path} 가 존재하지 않습니다. if_count 벡터라이저를 생성합니다..")
        vectorizer = CountVectorizer(
                    max_features=20000,
                    ngram_range=(1,1), 
                    min_df=3,            
                    max_df=0.95           
                    )
        vectorizer.fit(spam_df['message'])
        joblib.dump(vectorizer, vectorizer_path)
        print("vectorizer 생성 및 저장")
        
    
    
    # 벡터화
    X_spam = vectorizer.transform(spam_df['message'])

    #find_optimal_k(X_spam[:10000], max_k=8)
    #k=2가 제일 좋은 성능 보였음..
    #return


    # KMeans 학습
    sample_size = min(50_000, X_spam.shape[0])
    print(f"📌 KMeans 클러스터링 시작: {num_clusters}개 클러스터 (샘플 {sample_size}개)")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_spam[:sample_size])

    # 전체 스팸 데이터에 클러스터 ID 부여
    spam_clusters = kmeans.predict(X_spam)
    spam_df['category'] = spam_clusters

    # 저장 여부 선택
    while True:
        try:
            flag = int(input("KMeans 모델과 카테고리 결과를 저장할까요? 1: 예 / 0: 아니오 "))
            if flag == 1:
                # 카테고리 붙은 CSV (선택) 저장
                new_csv_path = if_phishing_path.replace('.csv', '_with_category.csv')
                spam_df.to_csv(new_csv_path, index=False, columns=['message', 'category'], encoding='utf-8-sig')
                # KMeans 모델 저장
                joblib.dump(kmeans, kmeans_model_path)
                print(f"저장 완료: {new_csv_path}, {kmeans_model_path}")
                break
            elif flag == 0:
                print("저장하지 않고 종료합니다.")
                break
            else:
                print("0 또는 1을 입력해주세요.")
        except ValueError:
            print("잘못된 입력입니다.")



def if_train_and_evaluate_model(
    category_phishing_path='data/processed/if_cleand_data_with_category.csv',
    vectorizer_path='model/if_count_vectorizer.pkl',
    kmeans_model_path='model/kmeans_spam.pkl',
    if_model_dir='model/category_if_models/',
    contamination=0.01,
    min_samples_per_category=500,  # ✅ 소형 기준
    merged_category_id=999         # ✅ 기타로 병합할 ID
):

    os.makedirs(if_model_dir, exist_ok=True)

    # 1) 카테고리 포함 스팸 데이터 로드
    if os.path.exists(category_phishing_path):
        spam_df = pd.read_csv(category_phishing_path, encoding='utf-8-sig')
        print(f"📌 카테고리 포함 스팸 로드 완료: {len(spam_df)} rows")
    else:
        print(f"{category_phishing_path} 가 존재하지 않습니다.")
        return

    # 2) Vectorizer 로드
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        print(f"{vectorizer_path} 가 존재하지 않습니다.")
        return

    # 3) KMeans 로드
    if os.path.exists(kmeans_model_path):
        kmeans = joblib.load(kmeans_model_path)
    else:
        print(f"{kmeans_model_path} 가 존재하지 않습니다.")
        return

    # 4) 벡터화
    X_spam = vectorizer.transform(spam_df['message'])

    # 5) category 없으면 KMeans로 예측
    if 'category' not in spam_df.columns:
        spam_df['category'] = kmeans.predict(X_spam)
        print(f"📌 KMeans로 category 컬럼 추가 완료")

    # 6) 소형 클러스터 식별 & 병합
    small_clusters = []
    merged_indices = []

    for category_id in spam_df['category'].unique():
        n_samples = (spam_df['category'] == category_id).sum()
        if n_samples < min_samples_per_category:
            small_clusters.append(category_id)
            merged_indices.extend(spam_df[spam_df['category'] == category_id].index.tolist())

    if merged_indices:
        spam_df.loc[merged_indices, 'category'] = merged_category_id
        print(f"⚙️ 소형 클러스터 {small_clusters} 병합 → '기타' Category {merged_category_id} (총 {len(merged_indices)} samples)")
    else:
        print("✅ 소형 클러스터 없음 → 병합 생략")

    # 7) 카테고리별 IF 학습
    if_models = {}
    score_ranges = {}  # 카테고리별 score 범위 기록
    for category_id in spam_df['category'].unique():
        X_cat = X_spam[spam_df['category'] == category_id]
        n_samples = X_cat.shape[0]

        if n_samples < min_samples_per_category:
            print(f"⚠️ Category {category_id} : {n_samples}개 → 너무 적어 IF 스킵")
            continue

        print(f"📌 Category {category_id} : {n_samples}개 → IF 학습")
        if_model = IsolationForest(contamination=contamination, random_state=42)
        if_model.fit(X_cat)
        if_models[category_id] = if_model

        # 학습 데이터에서 decision_function 점수 범위 확인
        try:
            scores = if_model.decision_function(X_cat)
            score_min, score_max = scores.min(), scores.max()
            print(f"   ┗ decision_function score: min={score_min:.4f}, max={score_max:.4f}")
            score_ranges[category_id] = {
            "min": score_min,
            "max": score_max
            }
        except Exception as e:
            print(f"   ⚠️ 스코어 계산 오류: {e}")

    # ✅ 8) 저장 여부
    while True:
        try:
            flag = int(input("score_range 및 카테고리별 IF 모델 저장할까요? 1: 예 / 0: 아니오 ").strip())
            if flag == 1:
                for cid, if_model in if_models.items():
                    path = os.path.join(if_model_dir, f"if_category_{cid}.pkl")
                    joblib.dump(if_model, path)
                    print(f"✅ 저장 완료: {path}")
                ranges_path = os.path.join(if_model_dir, "if_score_ranges.pkl")
                joblib.dump(score_ranges, ranges_path)
                print(f"✅ score_ranges 저장 완료: {ranges_path}")
                break
            elif flag == 0:
                print("저장을 건너뜁니다.")
                break
            else:
                print("0 또는 1을 입력해주세요.")
        except ValueError:
            print("잘못된 입력입니다.")






if __name__ == '__main__':
    #naive_train_and_evaluate_model()
    #rf_train_and_evaluate_model() 
    #k_means_clustering()
    if_train_and_evaluate_model()
