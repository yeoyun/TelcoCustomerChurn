# 📦 컬럼 인코딩 및 처리 모듈화 함수
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import joblib


def preprocess_telco_columns(df_raw: pd.DataFrame):
    df = df_raw.copy()

    df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})

    drop_cols = [
        'Customer ID', 'Country', 'State', 'City', 'Quarter',
        'Customer Status', 'Churn Label', 'Churn Score', 'Churn Category', 'Churn Reason'
    ]

    label_cols = [
        'Gender', 'Under 30', 'Senior Citizen', 'Married', 'Dependents',
        'Referred a Friend', 'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Unlimited Data', 'Paperless Billing'
    ]

    onehot_cols = ['Contract', 'Payment Method', 'Internet Type', 'Offer']

    # 🔹 Label Encoding (저장용)
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].fillna('Unknown'))

    df = pd.get_dummies(df, columns=onehot_cols, dummy_na=True)
    df = df.drop(columns=drop_cols, errors='ignore')

    return df, le


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 콘텐츠 스트리밍 개수
    df['StreamingCount'] = (
        df['Streaming TV'] + df['Streaming Movies'] + df['Streaming Music']
    )

    # 보안 서비스 활용 개수
    df['SecureUser'] = (
        df['Online Security'] + df['Device Protection Plan'] + df['Premium Tech Support']
    )

    # 가족 여부
    df['FamilyOriented'] = ((df['Married'] == 1) & (df['Dependents'] == 1)).astype(int)

    # 청년 1인가구 여부
    df['IsYoungAndAlone'] = ((df['Under 30'] == 1) & (df['Dependents'] == 0)).astype(int)

    return df


def scale_numeric_features(df: pd.DataFrame, numeric_cols: list):
    df = df.copy()
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler


# ✅ VIF 확인 함수
def calculate_vif_safe(df: pd.DataFrame) -> pd.DataFrame:
    # 🔹 숫자형 컬럼만 추출
    X = df.select_dtypes(include=['int64', 'float64', 'uint8']).copy()

    # 🔹 상수항 추가
    X = add_constant(X)

    # 🔹 VIF 계산
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

# ✅ VIF 제거 함수
def drop_high_vif(df: pd.DataFrame, threshold: float = 30.0) -> pd.DataFrame:
    df = df.copy()
    while True:
        vif_df = calculate_vif_safe(df)
        max_vif = vif_df.loc[vif_df['feature'] != 'const', 'VIF'].max()
        if max_vif > threshold:
            drop_feature = vif_df.loc[vif_df['feature'] != 'const'].sort_values('VIF', ascending=False).iloc[0]['feature']
            df = df.drop(columns=[drop_feature])
        else:
            break
    return df

def save_model_with_assets(model, features, threshold, scaler, label_encoder, path='xgb_telco_bundle.pkl'):
    package = {
        'model': model,
        'features': features,
        'threshold': threshold,
        'scaler': scaler,
        'label_encoder': label_encoder
    }
    joblib.dump(package, path)


# ✅ 모델 로드 함수
def load_model_with_assets(path='xgb_telco_bundle.pkl'):
    return joblib.load(path)

