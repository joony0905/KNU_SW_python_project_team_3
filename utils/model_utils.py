import numpy as np
from scipy.sparse import hstack, csr_matrix
import joblib
import pandas as pd

def plot_calibration_curve(model, X_test, y_test, model_name="model", n_bins=20):
        """
        모델의 예측 확률에 대한 calibration curve와 Brier score를 시각화합니다.
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import brier_score_loss
        
        # 예측 확률 가져오기
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]  # spam 확률
        else:
            raise ValueError("모델에 predict_proba 메서드가 없습니다.")

        # Calibration curve 계산
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)

        # Brier score 계산 (예측 확률과 실제 레이블 간의 평균 제곱 오차)
        brier = brier_score_loss(y_test, y_prob)

        # 시각화
        plt.figure(figsize=(6, 6))
        plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name} (Brier={brier:.4f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.title('Calibration Curve')
        plt.xlabel('Predicted Probability')
        plt.ylabel('True Fraction of Positives')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def load_model_and_vectorizer_Naive(vectorizer_path='model/nv_tf_idf_vectorizer.pkl', model_path='model/nv_spam_classifier.pkl'):
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except FileNotFoundError:
        raise FileNotFoundError("모델 또는 벡터라이저 파일을 찾을 수 없습니다.")

def load_model_and_vectorizer_Random(vectorizer_path='model/rf_count_vectorizer.pkl', model_path='model/rf_spam_classifier.pkl', category_columns_path='model/rf_category_columns.pkl'):
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        category_columns = joblib.load(category_columns_path)
        return vectorizer, model, category_columns
    except FileNotFoundError:
        raise FileNotFoundError("모델 또는 벡터라이저 파일을 찾을 수 없습니다.")
    
def rf_calculate_prob(message, vectorizer, model, category_columns):
    """
    CountVectorizer + log(1 + tf) + 단어 수 feature + softmax 후처리를 적용한 예측 함수.

    Parameters:
        message (str): 전처리된 텍스트
        vectorizer: CountVectorizer 객체 (사전 학습됨)
        model: 학습된 RandomForestClassifier
        category_columns (list): 카테고리 이름 리스트 (One-hot 순서)

    Returns:
        final_prob (float): Softmax 가중 평균 스팸 확률
    """

    # 1. 메시지 → 벡터화 (Count 기반)
    message_vec = vectorizer.transform([message])

    # 2. log(1 + tf) 변환
    message_log = message_vec.copy()
    message_log.data = np.log1p(message_log.data)

    # 3. 문장 길이 feature (단어 수)
    num_words = len(message.split())
    length_feature = csr_matrix([[num_words]])

    # 4. 카테고리별 확률 저장
    probs = []

    for cat in category_columns:
        # One-hot 벡터 생성
        cat_sparse = build_category_vector(cat, category_columns)

        # 전체 입력 구성: [log_count + length + category]
        combined = hstack([message_log, length_feature, cat_sparse])

        # 예측
        prob = model.predict_proba(combined)[0][1]
        probs.append(prob)

    weights = rf_softmax(probs)
    final_prob = np.sum(np.array(probs) * weights)

    return final_prob

def nb_softmax(x):
    """
    nb Softmax 확률 계산 함수.

    Parameters:
        x (array-like): 각 클래스에 대한 점수 (logits 또는 log 확률)

    Returns:
        probs (np.ndarray): softmax를 적용한 클래스별 확률 분포
        -> 점수들의 평균
    """
    exp_scores = np.exp(x - np.max(x))  # overflow 방지
    probs = exp_scores / exp_scores.sum()
    return probs

def rf_softmax(x):
    """
    Softmax 계산 함수 (랜덤포레스트 카테고리 후처리용).

    Parameters:
        x (array-like): 실수 값들의 리스트 (예: 카테고리별 예측 확률 등)

    Returns:
        np.ndarray: softmax를 적용한 결과값 (확률 분포)
        -> 카테고리별 확률로 가중 평균
    """
    e_x = np.exp(x - np.max(x))  # Overflow 방지를 위해 안정성 처리
    return e_x / e_x.sum()

def build_category_vector(category, category_columns):
    """
    주어진 카테고리에 대해 One-hot 벡터를 생성합니다.

    Parameters:
        category (str): 현재 메시지의 카테고리 (예: 'bank', 'chat', 'ad' 등)
        category_columns (list): 학습에 사용된 전체 카테고리 리스트

    Returns:
        csr_matrix: 해당 category에 대한 One-hot 인코딩 결과 (희소 행렬 형식)

    설명:
        - 학습에 사용된 category_columns 기준으로 One-hot vector를 만듭니다.
        - category가 category_columns에 포함되어 있으면 해당 위치만 1, 나머지는 0입니다.
        - category가 'ad'처럼 제외된 항목이면 아무 위치도 1이 되지 않아 전체가 0인 벡터가 생성됩니다.
        - 이 구조 덕분에 'ad' 카테고리는 텍스트+길이 벡터만으로 추론하게 됩니다.
        - ad 카테고리는 문자 종류 분류 없이 광고 문자들 총망라
    """
    onehot = pd.DataFrame([[0] * len(category_columns)], columns=category_columns)
    
    if category in category_columns:
        onehot[category] = 1  # ad는 category_columns에 없음 → 자동 제외

    return csr_matrix(onehot.values)

def get_naive_bayes_log_probs(processed_message, vectorizer, model, top_n=8):
    """
    Naive Bayes 모델을 사용해 로그확률 기반 스팸 확률 및 중요 단어 추출.

    Parameters:
        processed_message (str): 전처리된 텍스트
        vectorizer: 사전 학습된 CountVectorizer
        model: 학습된 Naive Bayes 모델
        top_n (int): 중요 단어 추출 개수

    Returns:
        spam_prob (float): 스팸일 확률
        important_words (list): 중요 단어 리스트
        message_vec: 전처리된 텍스트의 벡터 
    """

    message_vec = vectorizer.transform([processed_message])
    log_probs = model.feature_log_prob_
    indices = message_vec.indices
    counts = message_vec.data

    # 클래스별 로그우도 계산
    class_log_likelihoods = []
    for class_idx in range(len(model.classes_)):
        log_likelihood = 0
        for i, idx in enumerate(indices):
            log_likelihood += counts[i] * log_probs[class_idx, idx]
        class_log_likelihoods.append(log_likelihood)

    length = np.sum(counts)
    if length == 0:
        return None, None  # 입력 오류 처리용

    # 평균 로그우도 + 클래스 사전확률
    avg_log_likelihoods = np.array(class_log_likelihoods) / length
    class_log_prior = np.log(model.class_count_ / model.class_count_.sum())
    scores = class_log_prior + avg_log_likelihoods

    # softmax 변환
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    spam_prob = probs[model.classes_.tolist().index(1)]

    # 중요 단어 추출
    top_indices = indices[np.argsort(-counts)[:top_n]]
    important_words = [vectorizer.get_feature_names_out()[i] for i in top_indices]

    return spam_prob, important_words

