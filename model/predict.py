import sys 
import os
if __name__ == '__main__':
    ROOT_DIR = os.getcwd() 
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
from utils.preprocessing import preprocess_user_input
from utils.model_utils import *


def predict_with_selected_model(user_input, model_choice, return_label=False):
    # 선택된 모델에 따라 예측 수행
    if model_choice == 'naive_bayes':
        # 각각 모델로 예측
        vectorizer, model = load_model_and_vectorizer_Naive()
        return predict_message_naive(user_input, vectorizer, model)
    elif model_choice == 'random_forest':
            count_vec, rf_model, category_columns = load_model_and_vectorizer_Random()
            tfidf_vec, nb_model = load_model_and_vectorizer_Naive()
            if_count_vec, kmeans_model, if_models, range_score = load_kmeans_and_if_models()
            return predict_message_combined(user_input, count_vec, if_count_vec, tfidf_vec, rf_model, nb_model, kmeans_model, if_models, category_columns, range_score)
    else:
        return "잘못된 모델 선택"

def predict_message_naive(user_input, vectorizer, model):
    if vectorizer is None or model is None:
        return "모델 로드에 실패하였습니다. model/.pkl 파일을 확인해주세요.", 2

    if len(user_input) < 12:
        return "입력 텍스트가 너무 짧습니다.", 2

    processed_message = preprocess_user_input(user_input)

    spam_prob, important_words = get_naive_bayes_log_probs(processed_message, vectorizer, model)

    if spam_prob is None:
        return "해당 분류기는 한국어 기반으로 만들어졌습니다. 문장의 구성이 불필요한 언어(영어 등)가 너무 많은지 확인해주세요.", 2

    prediction_text = (
        #f"📌 예측 모델: Naive Bayes\n"
        f"- 이 메시지에서 주로 발견된 단어: {', '.join(important_words)} 등등.. \n"
        f"- 주로 출현되는 단어들을 기반으로 전문가가 스팸일 확률을 계산해봤어요!\n"
        f"- 스팸 확률은 {spam_prob:.2%}입니다!\n"
    )

    if spam_prob < 0.4:
        prediction_text += "✅ 이 문장은 정상일 가능성이 높습니다."
        label = "0"
    elif spam_prob < 0.5:
        prediction_text += "⚠️ 해당 문자는 스팸 문자일 경우가 의심됩니다. 주의를 요망합니다."
        label = "1"
    else:
        prediction_text += "🚨 다수의 단어들이 스팸에서 자주 보이는 패턴이에요. 스팸일 가능성이 높습니다!"
        label = "1"

    return prediction_text, label


def predict_message_combined(user_input, count_vec, if_count_vec, tfidf_vec, rf_model, nb_model, kmeans_model, if_models, category_columns, range_score):
    if count_vec and tfidf_vec and if_count_vec is None or rf_model and nb_model and kmeans_model and if_models is None:
        return "모델 및 벡터라이저 로드에 실패하였습니다. model/.pkl 파일을 확인해주세요.", 2

    if len(user_input) < 12:
        return "입력 텍스트가 너무 짧습니다.", 2

    processed_message = preprocess_user_input(user_input)

    rf_prob = rf_calculate_prob(processed_message, count_vec, rf_model, category_columns)
    nb_prob, _ = get_naive_bayes_log_probs(processed_message, tfidf_vec, nb_model)

    #입력 문장 count 벡터화
    X_input = if_count_vec.transform([processed_message])
    
    # 카테고리 추정
    category = kmeans_model.predict(X_input)[0]
    
    if_score = if_models[category].decision_function(X_input)[0]
    
    length_feature = len(processed_message.split())

    # scale 파라미터: 조정 가능
    scale = 0.1

    length_penalty = 1 - np.exp(-scale * length_feature)

    # 안전 clip
    length_penalty = min(max(length_penalty, 0.0), 1.0)
    
    range_info = range_score[category]
    score_min = range_info["min"]
    score_max = range_info["max"]

    # 카테고리별 맞춤형 정규화
    if_prob = (if_score - score_min) / (score_max - score_min)

    # 안전 clip
    if_prob = min(max(if_prob, 0.0), 1.0)   

    if_prob *= length_penalty

    # ✅ 최종 출력 메시지 생성
    prediction_text = (
        #f"📌 예측 모델: Naive Bayse + Random Forest + Isolation Forest\n"
        f"🤖 이 메시지는 AI 3중 검사 시스템으로 정밀 분석되었습니다!\n"
        "🔑 단어 전문가 + 다수결 심사 + 이상 패턴 탐지를 더한 분석 보고서를 작성해드리니 잘 읽어보세요!\n"
        f"🗒️ 단어 기반 스팸 확률:{nb_prob:.2%}\n"
        f"🧭 패턴 기반 스팸 확률:{rf_prob:.2%}\n"
        f"🧬 기존 패턴 유사 확률:{if_prob:.2%}\n"
    )
    prediction_text += get_final_spam_message(nb_prob, rf_prob, if_prob)

    for t in prediction_text:
        if t == "✅":
            label = 0
        else:
            label = 1
    
    return prediction_text, label


if __name__ == '__main__':
    user_input = input("문자를 입력하세요: ")
    processed_input = preprocess_user_input(user_input)
    print("입력 텍스트:", user_input)
    print("전처리 결과:", processed_input)
    #text,label = predict_with_selected_model(user_input, model_choice= "naive_bayes")
    #print(text)
    text,label = predict_with_selected_model(user_input, model_choice= "random_forest")
    print(text)