# KNU_SW_python_project_team_3
## 경북대학교 소프트웨어학과 파이썬기반 빅데이터 분석 프로젝트 3조
## 스팸 분류기 웹 애플리케이션 프로젝트

##   1. 프로젝트 개요
   본 프로젝트는 텍스트 메시지를 스팸인지 아닌지 분류하는 웹 애플리케이션을 구현합니다.<br>
   스팸 분류를 위해 Naive Bayes 모델과 Random Forest 모델을 사용하였으며,<br> 
   Flask를 사용하여 사용자 친화적인 웹 인터페이스를 제공합니다.<br> 
   사용자는 텍스트 메시지를 입력하고 입력된 메시지가 스팸일 가능성에 대한 예측을 받을 수 있습니다.
   * 왜 Naive Bayse와 Random Forest인가요? <br>
     ▶ 이진 분류 (정상/스팸)와 text문자의 특성상 두 모델이 제일 적합하다고 생각했습니다.
     
## 2. 25_05_08 초기 구성한 Flow-chart <br> (25_05_18 기준 웹 애플리케이션까지 확장 되었음) 

<img src="https://github.com/user-attachments/assets/c74900f3-7898-4c44-9792-be8b6f32d952" width="370" height="500">

##   2. 주요 기능


   * **2.1. 웹 인터페이스:** Flask를 사용하여 구축된 사용자 친화적인 웹 인터페이스를 통해 사용자는 스팸 분류기와 쉽게 상호 작용할 수 있습니다. <br><br>
   
   * **2.2. 모델 선택:** 사용자는 분류를 위해 나이브 베이즈 및 랜덤 포레스트 모델 중에서 선택할 수 있습니다.<br><br>
     **···** 🏃‍♀️ 25_05_18 기준 랜덤 포레스트 구현은 진행 중 <br><br>
     
   * **2.3. 텍스트 전처리:** 입력 텍스트는 다음과 같은 한국어에 특화된 기술을 사용하여 전처리됩니다. <br>
        * 2.3.1. 클리닝 (한국어 이외의 문자 및 특정 패턴 제거)  **···** ✅25_05_09 완료 <br>
        * 2.3.2. 토큰화 (konlpy의 Okt 사용)  **···** ✅25_05_11 완료 <br>
        * 2.3.3. 불용어 제거 및 형태소 단위로 분리 **···** ✅25_05_11 완료 <br> (stopwords.txt -> https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a 를 참조했습니다.)<br><br>
        
   * **2.4. 나이브 베이즈 분류:** 나이브 베이즈 모델이 스팸 분류에 사용됩니다. 구현에는 다음이 포함됩니다.
        * 2.4.1. 텍스트 데이터의 TF-IDF 벡터화 **···** ✅25_05_09 완료
        * 2.4.2. 로그 확률 계산 및 정규화  **···** ✅25_05_17 완료 <br>
        (확률곱의 경우 문장이 길어질 경우 많은 단어의 출현으로 인해 편향된 확률을 나타냄을 확인하여, <br> [로그 확률 / 문장 길이]로 정규화 하는 작업을 진행하였습니다.) <br><br>

   * **2.5. 랜덤 포레스트 분류:** 랜덤 포레스트 모델이 스팸 분류에 사용됩니다. <br> **···** 🏃‍♀️ 25_05_18 기준 진행중 <br><br>
   나이브 베이즈 모델의 특성상, <br> 단어만으로 스팸인지 아닌지 유추하기에 정상 광고 문자와 스팸 광고 문자를 분류함에 있어 <br>아무리 튜닝을 잘해보아도 어려움을 느끼는 듯 하였습니다. <br> 따라서 랜덤포레스트를 추가하여 두 모델을 비교하고자 하였습니다.
        * 2.5.1. 텍스트 데이터의 TF-IDF 벡터화 **···** ✅25_05_09 완료
        * 2.5.2. 정상 문자 데이터 인코딩 (One_Hot Encoding) **···** 🏃‍♀️ 25_05_18 기준 진행중
        * 2.5.3. **···** 🏃‍♀️ 25_05_18 기준 진행중
   
   * **2.6. 모듈식 설계:** 프로젝트는 다음과 같이 디렉토리를 분리하는 모듈식 설계를 구현하여 프로젝트의 가독성 및 협업 능력을 증대하였습니다. <br> ✅25_05_18 완료
        * 2.6.1. 웹 애플리케이션 로직 (`app/`)
        * 2.6.2. 모델 학습 및 선택 (`model/`)
        * 2.6.3. 데이터 처리 (`data/`)
        * 2.6.4. 유틸리티 함수 (`utils/`)

##   3. 파일 구조
``` File Directory
KNU_SW_python_project_team_3/
├── app/                 # 웹 애플리케이션 관련 파일
│   ├── templates/         # HTML 템플릿
│   │   └── index.html     # 사용자 입력 폼 및 결과 표시
│   ├── static/            # 정적 파일 (CSS, JS)
│   │   ├── style.css
│   │   └── script.js
│   └── main.py            # Flask 웹 서버 진입점
│
├── model/               # 모델 학습 및 저장
│   ├── train_model.py      # 학습 스크립트 (나이브 베이즈, 랜덤 포레스트)
│   ├── data_loader.py      # 원본 데이터 병합 및 로드
├   ├── predict.py          # 학습 모델을 바탕으로 예측 실행 
│   ├── spam_classifier.pkl # 학습된 나이브 베이즈 모델
│   └── vectorizer.pkl      # TF-IDF 벡터화기
│
├── data/                # 데이터 및 전처리 파일
│   ├── raw/               # 원본 데이터
│   │   ├── init_phishing.csv
│   │   ├── comments_1.csv
│   │   └── generated_30k_ad_messages.csv
│   ├── processed/         # 전처리된 데이터
│   │   └── cleaned_data.csv
│   └── stopwords.txt      # 사용자 정의 불용어 목록
│
├── utils/               # 유틸리티 함수
│   ├── preprocessing.py   # 전처리 함수 (토큰화 등)
│   ├── evaluation.py      # 평가 지표 (정확도 등)
│   └── file_io.py         # 파일 로드/저장 함수
│
├── README.md            # 프로젝트 설명
├── requirements.txt     # 종속성 목록
└── run.py               # 프로젝트 실행 스크립트
```

##   4. 주요 사용 third party library 라이브러리
* Flask==2.3.3
* gunicorn==21.2.0
* pandas==2.2.1
* numpy==1.26.4
* scikit-learn==1.4.2
* konlpy==0.6.0
* JPype1==1.5.0
* swifter==1.4.0
* joblib==1.4.0
* jupyterlab==4.1.5
* matplotlib==3.8.3
* matplotlib-inline==0.1.7 <br><br>
`requirements.txt` 에 포함되어있습니다.

##   5. 설치 및 사용법

1.  **5.1. 리포지토리 복제:**

    ```bash
    cd 저장 원하는 파일 경로
    git clone https://github.com/joony0905/KNU_SW_python_project_team_3.git 
    ```

2.  **5.2. 종속성 설치:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **5.3. 데이터 및 모델 확인:**

    * 5.3.1. 필요한 데이터 파일이 `data/raw/` 및 `data/processed/` 디렉토리에 있는지, 저장되는지 확인합니다. <br> (`cleand_data`는 없는 경우 전처리하여 생성 됨.) <br><br> 
    * 5.3.2. 학습된 모델 (`spam_classifier.pkl`) 및 벡터화기 (`vectorizer.pkl`)가 `model/` 디렉토리에 있는지 확인합니다. <br> 없는 경우 `model/train_model.py`를 Run해 모델을 학습하여 생성합니다.

4.  **5.4. 애플리케이션 실행:**

    ```bash
    python run.py
    ```

5.  **5.5. 웹 애플리케이션 접속:**

    * 웹 브라우저를 열고 제공된 URL로 이동합니다 (일반적으로 `http://127.0.0.1:5000/`).

6.  **5.6. 분류기 사용:**

    * 입력 폼에 텍스트 메시지를 입력하고 모델을 선택합니다.
    * "분류" 버튼을 클릭하여 스팸 예측을 받습니다.

##   6. 코드 주요 사항

* **6.1. `app/main.py`:** Flask 라우트, 사용자 입력 처리, 모델 선택 및 예측 표시를 처리합니다.
* **6.2. `model/model_selector.py`:** 사용자 선택에 따라 적절한 모델을 로드하고 선택합니다.
* **6.3. `utils/preprocessing.py`:** 텍스트 클리닝, 토큰화 및 불용어 제거를 위한 함수를 포함합니다.
* **6.4. `predict_message_naive_bayes` (in `app/main.py`):** 나이브 베이즈에 특화되어 예측을 위해 로그 확률을 계산하고 정규화합니다.
* **6.5. 추후 추가 예정... 

##   7. 향후 개선 사항 (25_05_18 기준)

* 7.1. 랜덤 포레스트 모델 예측 구현
* 7.2. 더 나은 스타일링 및 반응형 디자인으로 사용자 인터페이스 개선
