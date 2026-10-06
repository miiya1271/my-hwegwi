# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 예측기 & 모델 평가", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연도별 기온 예측기 및 머신러닝 모델 평가")
st.markdown("""
과거 학습 기간(**최근 50년: 1956~2005년** vs **최근 100년: 1906~2005년**)에 따른 회귀선의 기울기 변화와,  
공통 **테스트 데이터(최근 20년: 2006~2025년)**에 대한 예측 성능(**MAE, MSE, R²**)을 비교합니다.
""")

# 1. 데이터 불러오기 및 전처리
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 연도별 관측일수 및 평균기온 계산
    yearly_df = df.groupby("연도")["평균기온"].agg(
        관측일수="count",
        연평균기온="mean"
    ).reset_index()
    
    # 2025년 이하 & 관측일수 300일 이상 조건 필터링
    filtered_df = yearly_df[
        (yearly_df["연도"] <= 2025) & 
        (yearly_df["관측일수"] >= 300)
    ].copy()
    
    return filtered_df

df_filtered = load_and_process_data()

# 2. 데이터 분할 (Train / Test)
train_50 = df_filtered[(df_filtered["연도"] >= 1956) & (df_filtered["연도"] <= 2005)]
train_100 = df_filtered[(df_filtered["연도"] >= 1906) & (df_filtered["연도"] <= 2005)]
test_data = df_filtered[(df_filtered["연도"] >= 2006) & (df_filtered["연도"] <= 2025)]
train_all = df_filtered[df_filtered["연도"] <= 2025]

X_test = test_data[["연도"]]
y_test = test_data["연평균기온"]

# 3. 모델 학습 및 예측
# (1) 최근 50년 학습 모델 (1956~2005)
model_50 = LinearRegression()
model_50.fit(train_50[["연도"]], train_50["연평균기온"])
pred_50 = model_50.predict(X_test)

# (2) 최근 100년 학습 모델 (1906~2005)
model_100 = LinearRegression()
model_100.fit(train_100[["연도"]], train_100["연평균기온"])
pred_100 = model_100.predict(X_test)

# (3) 전체 데이터 학습 모델 (전체 기간)
model_all = LinearRegression()
model_all.fit(train_all[["연도"]], train_all["연평균기온"])
pred_all = model_all.predict(X_test)

# 4. 성능 평가 측정 함수
def eval_metrics(model, y_true, y_pred):
    slope = model.coef_[0]
    intercept = model.intercept_
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rate_100 = slope * 10
