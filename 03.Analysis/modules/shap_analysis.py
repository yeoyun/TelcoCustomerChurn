# modules/shap_analysis.py

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance

def run_shap_analysis(model, X_test, top_n=20, show_plot=True):
    """
    SHAP 및 XGBoost 중요도 분석 결과를 비교하고 시각화합니다.

    Parameters:
    - model: 학습된 XGBoost 모델
    - X_test: 테스트 데이터셋 (DataFrame)
    - top_n: 상위 중요 변수 개수
    - show_plot: Summary plot 출력 여부

    Returns:
    - importance_df: SHAP 및 XGBoost 기반 중요도 비교 DataFrame
    """

    # ✅ 1. SHAP 분석
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP 평균 절댓값 중요도 계산
    shap_importance = np.abs(shap_values).mean(axis=0)
    shap_summary = pd.Series(shap_importance, index=X_test.columns)
    shap_top = shap_summary.sort_values(ascending=False).head(top_n)

    # ✅ 2. XGBoost Gain 기준 중요도
    xgb_gain = model.get_booster().get_score(importance_type='gain')
    xgb_gain_series = pd.Series(xgb_gain).sort_values(ascending=False)
    xgb_top = xgb_gain_series.head(top_n)

    # ✅ 3. 중요도 병합
    importance_df = pd.DataFrame({
        'SHAP Importance': shap_top,
        'XGBoost Gain': xgb_top
    }).fillna(0)

    importance_df = importance_df.sort_values(by='SHAP Importance', ascending=False)

    # ✅ 4. 시각화 (옵션)
    if show_plot:
        print("\nSHAP Summary Plot (bar)")
        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=top_n)
        print("\nSHAP Summary Plot (dot)")
        shap.summary_plot(shap_values, X_test, max_display=top_n)

        print("\nXGBoost Gain 기반 Feature Importance")
        plt.figure()
        plot_importance(
            model,
            importance_type='gain',
            max_num_features=top_n,
            height=0.4,
            grid=True,
            show_values=False
        )
        plt.title("XGBoost Feature Importance (Gain 기준)")
        plt.tight_layout()
        plt.show()

    return importance_df
