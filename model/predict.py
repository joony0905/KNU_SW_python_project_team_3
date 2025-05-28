import sys 
import os
from utils.preprocessing import preprocess_user_input
from utils.model_utils import *

if __name__ == '__main__':
    ROOT_DIR = os.getcwd() 
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    

def predict_with_selected_model(user_input, model_choice, return_label=False):
    # 선택된 모델에 따라 예측 수행
    if model_choice == 'naive_bayes':
        # 각각 모델로 예측
        vectorizer, model = load_model_and_vectorizer_Naive()
        return predict_message_naive(user_input, vectorizer, model)
    elif model_choice == 'random_forest':
        vectorizer, model, category_columns = load_model_and_vectorizer_Random()
        return predict_message_random(user_input, vectorizer, model, category_columns)
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
        f"📌 예측 모델: Naive Bayes\n"
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


def predict_message_random(user_input, vectorizer, model, category_columns):
    if vectorizer is None or model is None:
        return "모델 로드에 실패하였습니다. model/.pkl 파일을 확인해주세요.", 2

    if len(user_input) < 12:
        return "입력 텍스트가 너무 짧습니다.", 2

    processed_message = preprocess_user_input(user_input)

    final_prob = rf_calculate_prob(processed_message, vectorizer, model, category_columns)
    
    #iso_final_prob, iso_mean_prob = rf_calculate_final_prob(processed_message, vectorizer, iso_model, category_columns)

    # ✅ 최종 출력 메시지 생성
    prediction_text = (
        f"📌 예측 모델: Random Forest\n"
        f"- 이 메시지는 여러 전문가들에게 물어보며 스팸 확률을 계산했어요.\n"
        f"- 그래서 계산된 스팸 확률은 {final_prob:.2%}입니다!\n"
    )


    # ✅ 확률 기반 판단 분기
    if final_prob < 0.4:
        prediction_text += "✅ 이 문장은 정상일 가능성이 높습니다."
        label = "0"
    elif final_prob < 0.5:
        prediction_text += "⚠️ 해당 문자는 스팸 문자일 경우가 의심됩니다. 주의를 요망합니다."
        label = "1"
    else:
        prediction_text += "🚨 많은 기준에서 스팸으로 판단되어 스팸일 가능성이 높습니다!"
        label = "1"

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