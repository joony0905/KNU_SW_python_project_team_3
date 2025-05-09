import pandas as pd

phishing_df = pd.read_csv('init_phishing.csv', usecols=["Spam message"], encoding='cp949')
normal_df = pd.read_csv('normal_sms_samples_flat.csv', encoding = 'cp949')

phishing_df['message'] = phishing_df['Spam message']
normal_df['message'] = normal_df['normal message']

#라벨 추가
phishing_df['label'] = 1
normal_df['label'] = 0

#하나의 data로 병합
combined_df = pd.concat([phishing_df, normal_df], ignore_index=True)

#전처리
combined_df['message'] = (
        combined_df['message']
        .astype(str)
        .str.replace(r'ifg@', '', regex=True)
        .str.replace(r'\*+', '', regex=True)
        .str.replace(r'[^가-힣a-zA-Z0-9\s]', '', regex=True)
        .str.strip()  # 공백 제거
    )

combined_df = combined_df[['message', 'label']]

from sklearn.model_selection import train_test_split

X = combined_df['message']
y = combined_df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=7000)  # 가장많이 출현된 상위 단어 7000개
X_train_vec = vectorizer.fit_transform(X_train)  #벡터화 문자 -> 숫자
X_test_vec = vectorizer.transform(X_test)

# 모델 학습
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 예측
y_pred = model.predict(X_test_vec)

# 평가
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

import re
def predict_message(message):
    # 입력 메시지를 벡터화
    message_vec = vectorizer.transform([message])
    
    # 예측
    prediction = model.predict(message_vec)
    
       # 확률 예측
    prob = model.predict_proba(message_vec)[0]  # [0]으로 첫 번째 샘플의 결과 추출
    prediction = model.predict(message_vec)[0]  # 예측 라벨 (0 또는 1)
    
    # 결과 해석
    result = "스팸 메시지입니다." if prediction == 1 else "정상 메시지입니다."
    spam_prob = prob[1]  # 클래스 1 (스팸)에 대한 확률
    
    return f"{result} (스팸 확률: {spam_prob:.2%})"

# 3. 사용자로부터 입력 받기
user_input = input("메시지를 입력하세요: ")

# 텍스트 전처리
user_input = (
    user_input
    .replace('ifg@', '')  # 'ifg@' 문자열 제거
    .strip()  # 양쪽 공백 제거
)

# 특수 문자 제거 (한글, 영문, 숫자, 공백만 남기기)
user_input = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', user_input)

# 예측 결과 출력
print(user_input)
print(predict_message(user_input))