# 📊 Telco Customer Churn Prediction & Retention Design

고객 이탈을 사전에 예측하고, 분석 기반 리텐션 기능을 설계하는 End-to-End 데이터 분석 프로젝트입니다.  
모델링 성능 향상뿐 아니라 **시각화 대시보드**, **사용자 흐름 설계**, **리텐션 기능 기획**까지 포함했습니다.

---

## 🧩 프로젝트 개요

- **데이터셋**: Telco Customer Churn Dataset (Kaggle, IBM)
- https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3
- **분석 환경**: Python (Anaconda Notebook)
- **주요 목표**:
  - 고객 이탈 여부 예측 분류 모델 개발 (F1 Score, AUC 평가)
  - SHAP 기반 이탈 요인 해석 및 주요 변수 도출
  - AARRR 프레임워크 기반 고객 행동 분석
  - 이탈 고위험 고객군 행동 특성 분석 → 리텐션 기능 기획

---

## 🔍 분석 흐름 요약

1. 데이터 전처리 및 파생변수 생성

2. EDA (교차 시각화, 요금제/계약/만족도/서비스 조합 분석)

3. 이탈 예측 분류 모델 (XGBoost, SMOTE, Threshold 튜닝)

4. 주요 변수 해석 (SHAP, Feature Importance)

5. 클러스터링 기반 행동 특성 분석 및 사용자 흐름 설계

6. AARRR 프레임워크 기반 고객 행동 분석

7. 리텐션 기능 기획 및 사용자 시나리오 설계

8. QA 테스트 시나리오 및 Tableau 시각화 대시보드

---

## 🧠 사용 기술

- **언어/환경**: Python, Jupyter Notebook, Tableau
- **분석 기법**: EDA, Classification (XGBoost), SHAP, 퍼널 분석(AARRR), 군집분석
- **시각화**: Seaborn, Matplotlib, Tableau
- **기획 도구**: 사용자 시나리오, 시퀀스 다이어그램, QA 테스트 시나리오

---

## ⚙️ 모델링 & 성능 (XGBoost + SMOTE)

| Metric         | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|----------------|---------|---------|-----------|---------------|
| Precision      | 0.97    | 0.95    | 0.96      | 0.97          |
| Recall         | 0.98    | 0.93    | 0.95      | 0.97          |
| F1-Score       | 0.98    | 0.94    | 0.96      | 0.97          |
| Support        | 1035    | 374     | 1409      | 1409          |
| **F1 Score**   | -       | -       | -         | **0.935**     |
| **AUC Score**  | -       | -       | -         | **0.993**     |

📌 **Confusion Matrix**

| 실제\예측 | 예측 0 | 예측 1 |
|-----------|--------|--------|
| **0**     | 1015   | 20     |
| **1**     | 28     | 346    |

> 💡 성능 개선 전략:  
> - 클래스 불균형 보정
> - Threshold 튜닝을 통한 F1 최적화  
> - SHAP 기반 파생 변수 설계

---

## 🗂️ 리포지토리 구조

```
TelcoCustomerChurn/
│
├── data/                 # 원본 및 가공 데이터
├── notebooks/            # 분석 노트북 (EDA, 모델링, SHAP, AARRR 등)
├── modules/              # 전처리/모델링 함수 모듈화
├── outputs/              # 모델 성능 및 시각화 이미지
├── visualization/        # Tableau 시각화 자료
└── README.md             # 프로젝트 개요 및 설명
```

---

태블로 대시보드 : https://public.tableau.com/app/profile/.84698697/viz/TelecomChurnAnalysis_AARRR/2
