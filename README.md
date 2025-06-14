# KNU_SW_python_project_team_3
## 경북대학교 소프트웨어학과 파이썬기반 빅데이터 분석 프로젝트 3조
## 스팸 분류기 웹 애플리케이션 프로젝트

##   1. 프로젝트 개요
   본 프로젝트는 텍스트 메시지를 스팸인지 아닌지 분류하는 웹 애플리케이션을 구현합니다!<br>
   스팸 분류를 위해 Naive Bayes 모델과 Random Forest 모델을 사용하였으며,<br> 
   Flask를 사용하여 사용자 친화적인 웹 인터페이스를 제공합니다.<br> 
   사용자는 텍스트 메시지를 입력하고 입력된 메시지가 스팸일 가능성에 대한 예측을 받고 응답을 남길 수 있습니다.
     
## 2. 25_05_08 초기 구성한 Flow-chart <br> (25_05_18 기준 웹 애플리케이션까지 확장 되었음) 

<img src="https://github.com/user-attachments/assets/c74900f3-7898-4c44-9792-be8b6f32d952" width="370" height="500">

## 3. 실행 화면
<img src= "https://github.com/user-attachments/assets/19681257-ade1-4b71-a37d-1381bf793e81" width="840" height = "410">

<img src= "https://github.com/user-attachments/assets/86db60d2-17c2-444d-81f1-b70480730372" width="840" height = "410">

<img src= "https://github.com/user-attachments/assets/19f032ac-f533-4387-a2d0-e033b30dfd0f" width="840" height = "410">





##   4. 주요 기능


   * **4.1. 웹 인터페이스:** Flask를 사용하여 구축된 사용자 친화적인 웹 인터페이스를 통해 사용자는 스팸 분류기와 쉽게 상호 작용할 수 있습니다. <br><br>
   
   * **4.2. 모델 선택:** 사용자는 분류를 위해 **빠른 판단(나이브 베이즈)** 및 **정밀 판단(랜덤 포레스트)** 중에서 선택할 수 있습니다.<br><br>
     
   * **4.3. 텍스트 전처리:** 입력 텍스트는 다음과 같은 한국어에 특화된 기술을 사용하여 전처리됩니다. <br>
        * 4.3.1. 클리닝 (한국어 이외의 문자 및 특정 패턴 제거)  **···** ✅25_05_09 완료 <br>
        * 4.3.2. 토큰화 (konlpy의 Okt 사용)  **···** ✅25_05_11 완료 <br>
        * 4.3.3. 불용어 제거 및 형태소 단위로 분리 **···** ✅25_05_11 완료 <br> (stopwords.txt -> https://gist.github.com/spikeekips/40eea22ef4a89f629abd87eed535ac6a 를 참조했습니다.)<br><br>
        
   * **4.4. 나이브 베이즈 분류:** 나이브 베이즈 모델이 스팸 분류에 사용됩니다. 구현에는 다음이 포함됩니다.
        * 4.4.1. 텍스트 데이터의 TF-IDF 벡터화 **···** ✅25_05_09 완료
        * 4.4.2. 로그 확률 계산 및 정규화  **···** ✅25_05_17 완료 <br>
          (문장이 길어질 경우 많은 단어의 출현으로 인해 편향된 확률을 나타냄을 확인하여, <br> 로그우도기반으로 확률 계산하여 보정 하는 작업을 진행하였습니다.) <br><br>

   * **4.5. 랜덤 포레스트 분류:** 랜덤 포레스트 모델이 스팸 분류에 사용됩니다. <br> **···** ✅ 25_05_28 기준 랜덤 포레스트 구현 마무리(Beta) <br><br>
   프로젝트 진행중 과연 나이브 베이즈 모델만으로 <br> 스팸 탐지가 잘 예측되는가? 라는 의문이 들어,<br> 서로 다른 관점에서 접근하는 랜덤 포레스트 모델을 추가하여 <br> 두 모델을 비교해보고자 하였습니다.
        * 4.5.1. 텍스트 데이터의 Count 벡터화 **···** ✅ 25_05_09 완료 <br>
        * 4.5.2. 정상 문자를 카테고리 별로 분류 후 One_Hot Encoding, hstack 결합 **···** ✅ 25_05_27 기준 완료 <br>
        * 4.5.3. 문장 길이 피처 추가 및 softmax로 확률 평탄화 ✅ 25_05_27 기준 완료 <br>
        * 4.5.4. 하이퍼파라미터 튜닝을 통해 과적합 현상 보완 ✅ 25_05_27 기준 완료 <br>
        (랜덤포레스트의 경우 모델이 너무 확신하는 경향··· 과적합 현상이 많이 발생하여 여러 시도들을 통해 평탄화 작업을 하였습니다.) <br>
   <br>

   * 🤦‍♀️ 시행착오들..     
     <details><summary>[>] </summary>

      ### 🚧 시행착오 1. 데이터 수집의 어려움
      
      - 정상 문자 데이터는 **개인정보 이슈**로 인해 수집이 매우 어려웠습니다.
      - 초기에는 팀원들이 보유한 일부 정상 문자를 바탕으로 데이터를 확보했으며,
      - 데이터 증강 기법으로 동의어 치환, 형태소 재조합 등을 통해 양을 보충했습니다.
      - 하지만, **정상 문자의 패턴이 실제보다 단순**하여:
        - 모델 정확도는 높았지만,
        - 실제 스팸 분류 성능은 좋지 않았습니다.
      - 프로젝트 기간이 1달로 데이터 수집 기간이 충분하지못해,
        - 데이터 확보보다는 **모델 성능 최적화**에 집중하였습니다.
      
      ---
      
      ### 🧠 시행착오 2. 문맥 파악의 한계
      
      - 처음에는 **단어 기반 이진 분류**로 간단히 해결될 것이라 생각했습니다.
      - 그러나 실험 중 **단어만으로는 문장을 제대로 파악하지 못함**을 깨달았습니다.
      - Naive Bayes, Random Forest 모두 문맥 이해에 한계가 있어:
        - **n-gram**을 활용하여 문맥 정보를 일부 보완해 보았으나,
        - 2-gram 이상의 성능이 떨어져 **1-gram만 사용**했습니다.
      - 추후에 팀원들중 만약 대학원 진학을 하신다면..
      - 문맥을 더 잘 파악할 수 있는 딥러닝 기반 모델을 도입해보면 어떨까 싶습니다.
      
      ---
      
      ### ⚠ 시행착오 3. 모델의 과적합 문제

      - 솔직히 말씀드리면 저희가 모델 튜닝을 잘한 것이 아니라, 정상 문자 데이터셋이 단순해서인지 매우 우수한(?) 모델이 되었습니다.
      - 실제로 학습한 데이터셋으로만 진행하면 전부 다 맞추기도 하더라구요. 하지만, 실전에서 진행 해보니 어느 정도 성능은 나오지만,
      - 학습 데이터만큼 나오지는 않았고, 정상문자를 스팸문자로 오인하는 확률도 너무 높게 나오게 됐습니다. <br><br>
     
      #### Naive Bayse

      ``` Classification Report:
      Accuracy: 0.9851648351648352
                 precision    recall  f1-score   support
   
                0       0.99      0.99      0.99       7600
                1       0.95      0.96      0.96       1500

      accuracy                               0.99      9100
      macro avg          0.97      0.97      0.97      9100
      weighted avg       0.99      0.99      0.99      9100
      ```
      
      <img src ="https://github.com/user-attachments/assets/4bd60e85-7bbc-4314-aa46-78a02710b010" width = "420" height="400"> <br><br>

      #### Random Forest (하이퍼 파라미터 조정으로 Acurracy 조정됨)
     
      - 처음에는 98%가 넘게 나왔는데, 하이퍼 파라미터 조정후 시각화를 통해 어느정도 조정을 이뤄냈습니다. 
      - 원핫 인코딩을 통해 카테고리 피처를 추가하여 Accuracy를 높이는 튜닝 작업은 성공적이었습니다! <br><br>
      
      **카테고리 적용전**
        
      ``` Classification Report:
      Accuracy: 0.9519131924614506

      
                    precision    recall  f1-score   support

                 0       0.92      1.00      0.96      4534
                 1       1.00      0.90      0.95      4221

      accuracy                               0.95      8755
      macro avg          0.96      0.95      0.95      8755
      weighted avg       0.96      0.95      0.95      8755
      ```
      <br>
      
      <img src="https://github.com/user-attachments/assets/256bc322-24c1-4b31-866c-b3244753142d" width ="420" height= "400"> <br><br>
      
      
      **카테고리 적용후**
         
      ``` Classification Report:
      Accuracy: 0.9750999428897773
      
                    precision    recall  f1-score   support
   
                 0       0.95      1.00      0.98      4534
                 1       1.00      0.95      0.97      4221

      accuracy                               0.98      8755
      macro avg          0.98      0.97      0.98      8755
      weighted avg       0.98      0.98      0.98      8755
      ```
      <br>
      
      <img src= "https://github.com/user-attachments/assets/36c6941c-9997-4b06-83dc-388e8e06f70e" width = "400" height = "400">
      

      #### 벡터라이저 max_df / min_df 제한
      - **너무 자주 등장하거나 너무 희귀한 단어**를 제거해 과적합을 줄였습니다.

      
      #### 확률 보정
      - **Naive Bayes**:
        - 로그우도 기반반 확률을 적용하여 확률 과잉 확신 현상을 보정했습니다.
      - **Random Forest**:
        - 앞서 그래프에서 보셨듯이, 특히 랜덤포레스트가 과적합이 너무 심했습니다.  
        - 카테고리별 확률을 활용해:
          - 산술 평균
          - 기하 평균
          - `(max + mean)` 등의 다양한 보정 시도 해보았으나,
        - 문장 길이(length) 특성을 추가하고
        - 각 카테고리별 확률에 **가중치**를 적용하여 보정한 방식이 가장 효과적이었습니다. 
      
      #### `CalibratedClassifierCV` 보정 모델
      - `isotonic`, `sigmoid` 기반 보정을 시도하여 `Calibration_Curve`곡선을 그려보았는데...
        - 나이브베이즈는 sigmoid를 진행해도 별 효과가없었고...
        - 랜덤포레스트는 isotonic까지 둘다 시각화 해본 결과 효과가 있는듯(!) 하였으나
        - 오히려 **보정 전보다 더 과적합되는 현상**이 나타나 제외했습니다. <br><br>

   
      <img src= "https://github.com/user-attachments/assets/d9841787-c3e7-4215-b89b-49a8cb911d2c" width = "370" height ="480">
      <img src= "https://github.com/user-attachments/assets/ae30ccc8-c632-428a-af60-318ea1ee0ccd" width = "370" height ="480">

      <img src= "https://github.com/user-attachments/assets/a3078449-07b9-4017-a01e-e31a142e8231" width = "370" height ="480">
      <img src= "https://github.com/user-attachments/assets/80df65db-9897-4b12-bb72-9b47fed48cf9" width = "370" height = "480">
      <img src = "https://github.com/user-attachments/assets/6d0cdbf2-481d-4f8b-9a30-f159a1df6d90" width = "370" height = "480"> <br><br>

   
      - 스팸 판단을 위한 **최적 threshold**를 찾기 위해:
        - `threshold = 0.1 ~ 0.9` 범위를 `0.01` 간격으로 변화시키며
        - `F1 score` 기준 최적값을 찾았고,
        - 결과적으로 **0.5가 가장 우수**한 성능을 보였습니다. (~~뻘짓..~~)
      
      #### 하이퍼파라미터 튜닝
      - Random Forest의:
        - `n_estimators`
        - `max_depth`
        - `min_samples_leaf` 등 주요 파라미터를 수작업으로 반복 조정했습니다.
      
      - +) 파라미터 최적화를 해주는 `GridSearchCV`가 있다는 것을 뒤늦게 깨달았습니다. 추후에 적용해볼까합니다.
      
      </details>
        <br>





##  5. 모듈식 설계 

- 프로젝트는 다음과 같이 디렉토리를 분리하는 모듈식 설계를 구현하여,
- 최대한 코드의 결합, 응집도를 낮추도록 노력하였고,
- 프로젝트의 가독성 및 협업 능력을 증대하였습니다.
- 또한 각 주요 폴더마다 branch를 만들어 PR하는 방식으로 진행하였습니다.
- ✅25_05_18 완료
  
        * 5.1. 웹 애플리케이션 로직 (`app/`)
        * 5.2. 모델 학습 및 선택 (`model/`)
        * 5.3. 데이터 처리 (`data/`)
        * 5.4. 유틸리티 함수 (`utils/`)

    <br>

 - 파일 구조
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
│   ├── nv_spam_classifier.pkl # 학습된 나이브 베이즈 모델
│   └── rf_spam_classifier.pkl # 학습된 랜덤포레스트 모델
│   └── nv_tf_idf_vectorizer.pkl      # TF-IDF 벡터화기
│   └── rf_count_vectorizer.pkl #Count 벡터화기
│   └── rf_categoty_columns.pkl #One-hot 인코딩 카테고리 파일
│
├── data/                # 데이터 및 전처리 파일
│   ├── raw/               # 원본 데이터
│   │   ├── init_phishing.csv
│   │   ├── comments_1.csv
│   │   ├── generated_30k_ad_messages.csv
│   │   └── 그외 등등...
│   │
│   ├── processed/         # 전처리된 데이터
│   │   └── cleaned_data.csv
│   │   └── rf_cleaned_data.csv
│   └── stopwords.txt      # 사용자 정의 불용어 목록
│
├── utils/               # 유틸리티 함수
│   ├── preprocessing.py   # 전처리 함수 (토큰화 등)
│   ├── model_utils.py     # 모델에 활용되는 함수 
│   ├── file_io.py         # 파일 로드/저장 함수
│   └── feedback.py        # 사용자 피드백 기능 함수  
│
├── README.md            # 프로젝트 설명
├── requirements.txt     # 종속성 목록
└── run.py               # 프로젝트 실행 스크립트
```

##   6. 주요 사용 third party library 라이브러리
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
* matplotlib-inline==0.1.7 
* scipy==1.15.3 <br><br>
`requirements.txt` 에 포함되어있습니다.

##   7. 설치 및 사용법

7.1.  **리포지토리 복제:**

   ```bash
    cd 저장 원하는 파일 경로
    git clone https://github.com/joony0905/KNU_SW_python_project_team_3.git 
   ```

7.2.  **종속성 설치:**

   ```bash
    pip install -r requirements.txt
   ```

7.3.  **데이터 및 모델 확인:**

   * 7.3.1. 필요한 데이터 파일이 `data/raw/` 및 `data/processed/` 디렉토리에 있는지, 저장되는지 확인합니다. <br> (`cleand_data`는 없는 경우 전처리하여 생성 됨.) <br><br> 
   * 7.3.2. 학습된 모델 (`spam_classifier.pkl`) 및 벡터화기 (`vectorizer.pkl`)가 `model/` 디렉토리에 있는지 확인합니다. <br> 없는 경우 `model/train_model.py`를 Run해 각 모델을 학습하여 생성합니다.

7.4.  **애플리케이션 실행:**

   ```bash
   python run.py
   ```

7.5.  **웹 애플리케이션 접속:**

   * 웹 브라우저를 열고 제공된 URL로 이동합니다 (일반적으로 `http://127.0.0.1:5000/`).

7.6.  **분류기 사용:**

   * 입력 폼에 텍스트 메시지를 입력하고 모델을 선택합니다.
   * "분류" 버튼을 클릭하여 스팸 예측을 받습니다.

7.7. **사용자 응답 평가:**

   * 모델이 판단한 결과에 대해 만족, 불만족 평가를 남길 수 있습니다.

##   8. 코드 주요 사항

**<details><summary> 📁 8.1. app </summary>**
   
   ```bash/
    app/main.py : Flask 라우트, 사용자 입력 처리, 모델 선택 및 예측 표시를 처리합니다.
    app/index, ctyle, script.. : 웹 브라우저 구성 파일입니다.
   ```

</details>
   
**<details><summary> 📁 8.2. model</summary>**

  ```bash/ 
  `[model/train_model.py]`:

   모델 학습을 위한 전처리, 벡터화, 학습, 평가, 저장 기능이 포함된 메인 파이프라인입니다.
  
     Naive Bayes
     - CSV 파일 존재 여부에 따라 전처리 생략 가능  
     - `joblib`을 통한 모델 및 벡터라이저 로딩/저장 지원  
     - 사용자 입력으로 벡터 생성 여부 및 모델 재학습 선택 가능
     - `ConfusionMatrixDisplay`를 통한 시각화 및 성능 평가 포함
       
     Random Forest 
     - One-hot Encoding된 `category` 정보를 Count_vectorizer와 결합  
     - 과적합 방지를 위한 하이퍼파라미터 수동 조정  
     - `ConfusionMatrixDisplay`를 통한 시각화 및 성능 평가 포함  

  `[model/predict.py]`  

    사용자로부터 입력받은 메시지를 전처리하고, 선택한 모델(Naive Bayes 또는 Random Forest)을 통해 예측을 수행하는 모듈입니다.

     - predict_with_selected_model():   
       사용자가 선택한 모델( naive_bayes 또는 random_forest )에 따라 적절한 예측 함수를 호출합니다.  
   
     - predict_message_naive():  
       - Naive Bayes 모델 기반 예측 함수  
       - 메시지를 벡터화 후 로그우도 기반 확률 계산  
       - 주요 단어 추출 및 예측 확률에 따라 직관적인 피드백 문장을 생성
   
     - predict_message_random():  
       - Random Forest 기반 예측 함수  
       - 카테고리별 One-hot 벡터 및 문장 길이 특징 추가  
       - 여러 카테고리의 확률을 softmax 가중 평균하여 최종 스팸 확률 도출  
       - 확률에 따라 직관적인 피드백 문장을 생성  
   
      * 입력값이 너무 짧은 경우, 또는 모델 로딩 실패 시 예외 처리를 포함하여 사용자에게 안내 메시지를 제공합니다.

  `[model/data_loader.py]`
      
    정상 메시지(주제별), 스팸 메시지를 각각 로딩하고 통합하는 기능 제공
  
     - 주제별 카테고리를 병합하고 라벨(`ham`/`spam`) 지정  
     - 정제된 구조로 학습용 DataFrame 반환  
  ```
</details>


**<details><summary> 📁 8.3. utils</summary>**

   ```bash/
    `[utils/preprocessing.py]` : 텍스트 클리닝, 토큰화 및 불용어 제거를 위한 함수를 포함합니다.
   
    `[utils/file_io.py]`  
   
     CSV 로드 및 저장을 안정적으로 수행하는 유틸리티 함수 모음입니다.  
       - `save_dataframe_to_csv`, `load_csv` 등 포함  
       - 경로 자동 생성 및 예외 처리 지원
       
    `[utils/model_utils.py]`
   
     모델 후처리, 확률 계산, 확률 보정, 주요 단어 추출 등 다양한 기능을 포함하는 유틸리티 모듈입니다.
   
        - plot_calibration_curve():  
             모델의 예측 확률에 대해 `Calibration Curve` 및 `Brier Score`를 시각화합니다.
        
        - load_model_and_vectorizer_Naive() / Random():  
             Naive Bayes 또는 Random Forest 모델과 벡터라이저를 로드합니다.
        
        - rf_calculate_prob():  
             Random Forest 모델에 대해 `문장 길이 + 카테고리 One-hot + log TF` 기반 확률 계산을 수행하고,  
             Softmax 기반으로 여러 카테고리의 확률을 조합합니다.
        
        - rf_softmax() / nb_softmax():  
             Softmax 함수 구현 (RandomForest/NaiveBayes용), overflow 안정성 처리 포함.
        
        - build_category_vector():  
             주어진 카테고리에 대해 One-hot 벡터를 생성하는 함수로,  
             모델 입력에 필요한 카테고리 정보를 희소행렬로 반환합니다.
        
        - get_naive_bayes_log_probs():  
             Naive Bayes 모델에서 사용된 단어의 로그우도 기반으로 스팸 확률 계산,  
             메시지 내에서 스팸 판단에 중요한 단어 상위 `top_n` 개 추출 기능 포함.
       
    `[utils/feedback.py]`  
     사용자로부터 받은 예측 피드백을 저장하고, 향후 모델 개선을 위한 데이터로 활용할 수 있도록 구조화된 JSON 형식으로 기록합니다.
   
     - save_feedback():  
       단순 메시지-라벨 구조의 피드백을 `feedback_data.json`에 한 줄씩 저장합니다.
   
     - process_feedback():  
       확장된 피드백 구조를 받아,  

       - `message` (입력 메시지)  
       - `feedback` (정답 여부: 예/아니오 등)  
       - `prediction` (예측 결과)  
       - `model` (사용된 모델명) 들을 포함한 JSON 객체를 파일에 저장합니다.  

       파일은 `data/raw/feedback_data.json` 경로에 누적되며, 추후 모델 재학습에 활용할 것입니다.
   ```
</details>

##   9. 향후 방향..

* 9.1. GridSearchCV로 Random Forest 하이퍼파라미터 튜닝
* 9.2. feedback을 통해 받은 클라이언트측 데이터를 통해 재학습 (⭐)
* 9.3. 문맥 인지를 위한 딥러닝 모델 추가
