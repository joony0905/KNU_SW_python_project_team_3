import joblib
import numpy as np
from app.utils.preprocessing import preprocess_user_input
import os

def load_model_and_vectorizer_Naive(vectorizer_path='app/model/vectorizer.pkl', model_path='app/model/spam_classifier.pkl'):  # 기본 경로 설정
    try:
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return vectorizer, model
    except FileNotFoundError:
        raise FileNotFoundError("모델 또는 벡터라이저 파일을 찾을 수 없습니다.")
    
def load_model_and_vectorizer_Random(): #vectorizer_path='app/model/vectorizer.pkl', model_path='app/model/spam_classifier.pkl'
    return  #미구현
    

def predict_with_selected_model(processed_input, model_choice, return_label=False):
    if model_choice == 'naive_bayes':
        vectorizer, model = load_model_and_vectorizer_Naive()
        prediction, label = predict_message_naive(processed_input, vectorizer, model)
    elif model_choice == 'random_forest':
        prediction, label = predict_message_random()
    else:
        return ("잘못된 모델 선택", "unknown") if return_label else "잘못된 모델 선택"

    return (prediction, label) if return_label else prediction



import numpy as np

def predict_message_naive(message, vectorizer, model):
    if vectorizer is None or model is None:
        return "모델 로드 실패. 예측 불가.", "unknown"

    if len(message) < 12:
        return "입력 텍스트가 너무 짧습니다.", "too_short"

    message_vec = vectorizer.transform([message])

    log_probs = model.feature_log_prob_
    indices = message_vec.indices
    counts = message_vec.data

    class_log_likelihoods = []
    for class_idx in range(len(model.classes_)):
        log_likelihood = 0
        for i, idx in enumerate(indices):
            log_likelihood += counts[i] * log_probs[class_idx, idx]
        class_log_likelihoods.append(log_likelihood)

    length = np.sum(counts)
    if length == 0:
        return "입력 텍스트가 너무 짧습니다.", "too_short"

    avg_log_likelihoods = np.array(class_log_likelihoods) / length
    class_log_prior = np.log(model.class_count_ / model.class_count_.sum())
    scores = class_log_prior + avg_log_likelihoods

    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / exp_scores.sum()

    spam_prob = probs[model.classes_.tolist().index(1)]

    if spam_prob < 0.4:
        prediction_text = f"스팸 확률 {spam_prob:.2%} \n해당 문자는 정상 문자일 확률이 높습니다."
        label = "0"
    elif spam_prob < 0.5:
        prediction_text = f"스팸 확률 {spam_prob:.2%} \n해당 문자는 스팸 문자일 경우가 의심됩니다. 주의를 요망합니다."
        label = "1"#의심문자도 스팸이라 보기
    else:
        prediction_text = f"스팸 확률 {spam_prob:.2%} \n해당 문자는 스팸 문자일 경우가 거의 확실합니다!!!"
        label = "1"

    return prediction_text, label
#미구현
def predict_message_random():
    return "랜덤 포레스트는 아직 미구현 되었습니다."

if __name__ == '__main__':
    user_input = input("문자를 입력하세요: ")
    processed_input = preprocess_user_input(user_input)
    print("입력 텍스트:", user_input)
    print("전처리 결과:", processed_input)
    vectorizer, model = predict_with_selected_model(processed_input, model_choice= "Naive Bayes")
