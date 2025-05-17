# csv 파일 불러오기, 한국어 불용어 리스트를 사용한 전처리, 학습, 실행(정상적으로 실행됨)
# 랜덤 포레스트 분류기 모델 
##pandas: 데이터프레임을 다루는 데 사용됩니다. CSV 파일을 읽고 데이터를 처리하는 데 사용됩니다.
##
##re: 정규 표현식을 사용하여 문자열에서 특정 패턴을 찾거나 변경하는 데 사용됩니다.
##
##sklearn.model_selection.train_test_split: 데이터를 학습용 데이터와 테스트용 데이터로 나누는 데 사용됩니다.
##
##sklearn.feature_extraction.text.TfidfVectorizer: 텍스트 데이터를 벡터화하는 데 사용됩니다. 텍스트 데이터를 수치화하여 머신러닝 모델에 입력할 수 있게 만듭니다.
##
##sklearn.ensemble.RandomForestClassifier: 랜덤 포레스트 분류기를 사용하여 텍스트 데이터를 분류합니다.
##
##sklearn.metrics.classification_report, accuracy_score: 모델의 성능을 평가하는 데 사용됩니다.
##
##konlpy.tag.Okt: 한국어 형태소 분석기인 Okt를 사용하여 텍스트를 토큰화하고 형태소 분석을 수행합니다.

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from konlpy.tag import Okt

# 한국어 불용어 리스트 정의
stop_words = {
    "의", "가", "이", "은", "는", "을", "를", "에", "와", "과", "한", "하다", "있다", "없다", 
    "되다", "되었", "되었는", "이다", "이다", "저", "그", "이", "저희", "그쪽", "그녀", 
    "그는", "너", "나", "대해", "또한", "자", "거기", "여기", "조금", "그래서", "그러면", 
    "그런", "같은", "등", "들", "좀", "많이", "제", "저기"
}

# 텍스트 전처리 함수
def preprocess_text(text):
    # 한글, 공백, 숫자만 남기고 모두 제거
    text = re.sub(r'[^가-힣\s]', '', text)
    
    # 불용어 제거
    okt = Okt()
    tokens = okt.morphs(text)  # 형태소 분석, 텍스트를 형태소 단위로 나눔

    # 불용어 목록에 있는 단어들은 필터링하여 제거하고, 의미 있는 단어들만 남깁니다.
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 결과적으로 불용어가 제거된 전처리된 텍스트를 반환합니다.
    return ' '.join(filtered_tokens)

# 1. CSV 파일 로드
data = pd.read_csv("spam_ham_messages.csv", encoding='cp949')

# 2. 'v1' 컬럼이 라벨(ham/spam), 'v2' 컬럼이 메시지 내용
X = data['v2']  # 메시지 내용
y = data['v1']  # 라벨 (ham 또는 spam)

# 3. 텍스트 전처리 적용
##  **apply(preprocess_text)**를 사용하여 X에 있는 모든 메시지에 대해 preprocess_text 함수를 적용합니다.
##  각 메시지는 전처리된 후 다시 X에 저장됩니다.
X = X.apply(preprocess_text)

# 4. 데이터 분리: 학습용 데이터와 테스트용 데이터 (80% 학습, 20% 테스트)
##**train_test_split**를 사용하여 데이터를 훈련 세트(80%)와 테스트 세트(20%)로 나눕니다.
##random_state=42는 랜덤하게 데이터를 분리할 때 동일한 분할을 재현할 수 있도록 하기 위해 설정합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 5. TF-IDF 벡터화기 생성
# TfidfVectorizer는 TF-IDF(Term Frequency-Inverse Document Frequency) 방식으로 텍스트를 수치 데이터로 변환합니다.
tfidf_vectorizer = TfidfVectorizer()

# 6. 학습 데이터와 테스트 데이터를 TF-IDF 벡터로 변환
##fit_transform은 훈련 데이터를 사용해 단어의 중요도를 학습하고, 데이터를 벡터로 변환합니다.
##transform은 학습된 벡터화기를 사용하여 테스트 데이터를 벡터화합니다.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 7. 랜덤 포레스트 분류기 모델 생성
# 랜덤 포레스트 분류기(RandomForestClassifier)를 사용하여 모델을 생성합니다. n_estimators=100은 100개의 결정 트리를 생성하여 앙상블 방식으로 분류합니다.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 8. 모델 학습
# model.fit은 훈련 데이터를 사용하여 모델을 학습시킵니다.
model.fit(X_train_tfidf, y_train)

# 9. 테스트 데이터에 대한 예측
# model.predict는 테스트 데이터를 입력받아 예측 결과를 생성합니다.
y_pred = model.predict(X_test_tfidf)

# 10. 정확도 출력
# **accuracy_score**는 예측된 값과 실제 값 사이의 정확도를 계산하여 출력합니다.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# 11. 분류 성능 평가 (Precision, Recall, F1-score 등)
##**classification_report**는 Precision, Recall, F1-score 등의 성능 지표를 계산하여 출력합니다.
##이 지표들은 모델이 얼마나 잘 분류했는지를 평가하는 데 유용합니다.
print(classification_report(y_test, y_pred))

# 12. 예시 메시지 예측 함수
##이 함수는 새로운 메시지가 입력되면:
## 1. 전처리 과정을 거친 후
## 2. 벡터화를 진행한 후
## 3. 학습된 랜덤 포레스트 모델을 사용하여 메시지가 spam인지 ham인지를 예측합니다.
def predict_spam_or_ham(message):
    # 입력된 메시지 전처리 및 벡터화
    message = preprocess_text(message)
    message_tfidf = tfidf_vectorizer.transform([message])
    
    # 예측
    prediction = model.predict(message_tfidf)
    
    return prediction[0]

# 13. 예시 메시지 예측
new_message = "[광고] 할인 이벤트 안내! 최대 50% 할인"
new_message_2 = "[Web발신]ifg@(광고)[KT] **월 *주차 WEEKLY BESTifg@ifg@이번 주 식품 BEST 아이템을 특별가로!?ifg@ifg@?최대 **% SALE?ifg@?RCB폴로클럽 남녀패딩 푸퍼 다운 점퍼ifg@?리앤쿡 노르딕 나이프 *종 세트ifg@?테팔 IH 매직핸즈 이모션 스텐 멀티 *P 세트ifg@?센소다인 오리지날 플러스 치약 ***g *개ifg@?플랜비안 카본 난연매트 온열 전기요ifg@ifg@초특가/*+*/추가 증정/추가 할인 등 다양한 혜택?ifg@ifg@?더 많은 상품 확인하기?ifg@https://su.kt.co.kr/AlVLjzGifg@ifg@?리뷰 쓰고 적립금받자!?ifg@https://su.kt.co.kr/AlVLjzoifg@ifg@▶이용 문의 : K-Deal 고객센터 ******** (평일 **:**~**:**) ifg@ifg@프리미엄 쇼핑 혜택 KT와 함께하세요!ifg@ifg@무료 수신거부: ***-***-****ifg@ifg@[KT]"
prediction = predict_spam_or_ham(new_message_2)
print(f"The message is: {prediction}")

