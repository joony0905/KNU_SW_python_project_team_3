import joblib
import numpy as np
from utils.preprocessing import preprocess_user_input
import os

def load_model_and_vectorizer_Naive(vectorizer_path='model/vectorizer.pkl', model_path='model/spam_classifier.pkl'):  # 기본 경로 설정
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except FileNotFoundError:
        raise FileNotFoundError("모델 또는 벡터라이저 파일을 찾을 수 없습니다.")
    
def load_model_and_vectorizer_Random(): #vectorizer_path='model/vectorizer.pkl', model_path='model/spam_classifier.pkl'
    return  #미구현
    

def predict_with_selected_model(processed_input, model_choice):
    # 선택된 모델에 따라 예측 수행
    if model_choice == 'naive_bayes':
        # Naive Bayes 모델로 예측
        vectorizer, model = load_model_and_vectorizer_Naive()
        return predict_message_naive(processed_input, vectorizer, model)
    elif model_choice == 'random_forest':
        #vectorizer, model = load_model_and_vectorizer_Random()
        return predict_message_random()
    else:
        return "잘못된 모델 선택"


def predict_message_naive(message, vectorizer, model):
    if vectorizer is None or model is None:
        return "모델 로드 실패. 예측 불가."

    if (len(message) < 12):
        return "입력 텍스트가 너무 짧습니다."

    message_vec = vectorizer.transform([message])

    log_probs = model.feature_log_prob_

    indices = message_vec.indices
    counts = message_vec.data

    class_log_likelihoods = []  #문장 길이가 긴 경우 단어가 많이 출력하면 확률곱이 늠. -> 로그 (확률) / 문장길이로 정규화함
    for class_idx in range(len(model.classes_)):
        log_likelihood = 0
        for i, idx in enumerate(indices):
            log_likelihood += counts[i] * log_probs[class_idx, idx]
        class_log_likelihoods.append(log_likelihood)

    length = np.sum(counts)
    if length == 0:
        return "입력 텍스트가 너무 짧습니다."

    avg_log_likelihoods = np.array(class_log_likelihoods) / length #정규화

    class_log_prior = np.log(model.class_count_ / model.class_count_.sum())

    scores = class_log_prior + avg_log_likelihoods

    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum() 

    spam_prob = probs[model.classes_.tolist().index(1)]
    print(f"")

    if (spam_prob < 0.4):
        return f"스팸 확률 {spam_prob:.2%} \n50 넘을시 1, 40~50구간 의심, 40미만 0\n해당 문자는 정상 문자일 확률이 높습니다."
    elif (spam_prob < 0.5):
        return f"스팸 확률 {spam_prob:.2%} \n50 넘을시 1, 40~50구간 의심, 40미만 0\n해당 문자는 스팸 문자일 경우가 의심됩니다. 주의를 요망합니다."
    else:
        return f"스팸 확률 {spam_prob:.2%} \n50 넘을시 1, 40~50구간 의심, 40미만 0\n해당 문자는 스팸 문자일 경우가 거의 확실합니다!!!"
    

#미구현
def predict_message_random():
    return "랜덤 포레스트는 아직 미구현 되었습니다."

if __name__ == '__main__':
    user_input = input("문자를 입력하세요: ")
    processed_input = preprocess_user_input(user_input)
    print("입력 텍스트:", user_input)
    print("전처리 결과:", processed_input)
    vectorizer, model = predict_with_selected_model(processed_input, model_choice= "Naive Bayes")