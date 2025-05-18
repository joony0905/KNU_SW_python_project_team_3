import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from konlpy.tag import Okt

# 1. 불용어 정의
stop_words = {
    "의", "가", "이", "은", "는", "을", "를", "에", "와", "과", "한", "하다", "있다", "없다", 
    "되다", "이다", "저", "그", "이", "저희", "그쪽", "그녀", "그는", "너", "나", 
    "대해", "또한", "자", "거기", "여기", "조금", "그래서", "그러면", "그런", 
    "같은", "등", "들", "좀", "많이", "제", "저기", "입니다"
}

okt = Okt()

# 2. 전처리 함수
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # 광고 태그 제거
    # text = re.sub(r'ifg@', ' ', text)
    text = text.replace('ifg@', 'ifg_at')  # 의미를 보존하면서 모델이 인식할 수 있게 유지
    text = re.sub(r'\[.*?\]', ' ', text)
    
    # 특수문자 제거 (한글/영문/공백 제외)
    text = re.sub(r'[^가-힣a-zA-Z\s]', ' ', text)
    
    # 형태소 분석
    tokens = okt.morphs(text, stem=True)
    
    # 불용어 제거
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    return ' '.join(tokens)

# 3. 데이터 불러오기
df = pd.read_csv('merged_ham_spam_dataset.csv', encoding='utf-8-sig')  # or utf-8-sig

# 4. 전처리 적용
df['v2'] = df['v2'].astype(str).apply(preprocess_text)

# 5. 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

# 6. TF-IDF 벡터화
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 7. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# 8. 예측 및 평가
y_pred = model.predict(X_test_tfidf)
print("정확도:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9. 새 메시지 판별 함수
def predict_spam_or_ham_with_proba(message):
    # 메시지 전처리
    processed = preprocess_text(message)
    vec = vectorizer.transform([processed])
    
    # 확률 예측
    proba = model.predict_proba(vec)[0]  # [spam 확률, ham 확률] 순서일 수 있음

    # 클래스 순서 확인
    classes = model.classes_  # 예: ['ham', 'spam']

    # 결과 추출
    spam_prob = proba[classes.tolist().index('spam')]
    ham_prob = proba[classes.tolist().index('ham')]

    # 최종 예측
    predicted = model.predict(vec)[0]

    print(f"📩 입력 메시지: {message}")
    print(f"🔍 예측 결과: {predicted.upper()}")
    print(f"📊 스팸 확률: {spam_prob * 100:.2f}%")
    print(f"📊 정상 확률: {ham_prob * 100:.2f}%")
    return predicted

msg = "[Web발신]ifg@한정 세일! 지금 클릭하면 50% 할인!"
predict_spam_or_ham_with_proba(msg)

