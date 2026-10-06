# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 예측기 & 모델 평가", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연도별 기온 예측기 & 모델 평가")

# 1. 데이터 불러오기 및 전처리
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
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

# 2. 데이터 세트 분할
# 훈련 데이터 세트
train_50 = df_filtered[(df_filtered["연도"] >= 1956) & (df_filtered["연도"] <= 2005)]
train_100 = df_filtered[(df_filtered["연도"] >= 1906) & (df_filtered["연도"] <= 2005)]
train_all = df_filtered[df_filtered["연도"] <= 2025]

# 공통 테스트 데이터 세트 (최근 20년: 2006~2025)
test_data = df_filtered[(df_filtered["연도"] >= 2006) & (df_filtered["연도"] <= 2025)]

X_test = test_data[["연도"]]
y_test = test_data["연평균기온"]

# 3. 모델 학습 및 평가
def train_and_eval(train_df, X_test, y_test):
    model = LinearRegression()
    model.fit(train_df[["연도"]], train_df["연평균기온"])
    
    slope = model.coef_[0]
    intercept = model.intercept_
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    rate_100 = slope * 100
    
    return {
        "model": model,
        "slope": slope,
        "intercept": intercept,
        "rate_100": rate_100,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "train_count": len(train_df)
    }

res_50 = train_and_eval(train_50, X_test, y_test)
res_100 = train_and_eval(train_100, X_test, y_test)
res_all = train_and_eval(train_all, X_test, y_test)

# ---------------------------------------------------------
# 1. 지표 카드 (기울기 및 테스트 성능 나란히 비교)
# ---------------------------------------------------------
st.subheader("📊 학습 모델별 기울기 및 테스트 데이터(2006~2025) 평가 결과")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔹 최근 50년 학습 (1956~2005)")
    st.metric("100년당 기온 상승 폭", f"+{res_50['rate_100']:.2f} °C")
    st.write(f"• **기울기**: `{res_50['slope']:.5f}` °C/년")
    st.write(f"• **회귀식**: $y = {res_50['slope']:.5f}x + ({res_50['intercept']:.3f})$")
    st.write(f"• **MAE**: `{res_50['mae']:.4f}` °C")
    st.write(f"• **MSE**: `{res_50['mse']:.4f}`")
    st.write(f"• **$R^2$**: `{res_50['r2']:.4f}`")

with col2:
    st.markdown("### 🔸 최근 100년 학습 (1906~2005)")
    diff_rate = res_100['rate_100'] - res_50['rate_100']
    st.metric("100년당 기온 상승 폭", f"+{res_100['rate_100']:.2f} °C", delta=f"{diff_rate:+.2f} °C (vs 50년)")
    st.write(f"• **기울기**: `{res_100['slope']:.5f}` °C/년")
    st.write(f"• **회귀식**: $y = {res_100['slope']:.5f}x + ({res_100['intercept']:.3f})$")
    st.write(f"• **MAE**: `{res_100['mae']:.4f}` °C")
    st.write(f"• **MSE**: `{res_100['mse']:.4f}`")
    st.write(f"• **$R^2$**: `{res_100['r2']:.4f}`")

with col3:
    st.markdown("### 🟢 전체 데이터 학습 (1908~2025)")
    st.metric("100년당 기온 상승 폭", f"+{res_all['rate_100']:.2f} °C")
    st.write(f"• **기울기**: `{res_all['slope']:.5f}` °C/년")
    st.write(f"• **회귀식**: $y = {res_all['slope']:.5f}x + ({res_all['intercept']:.3f})$")
    st.write(f"• **MAE**: `{res_all['mae']:.4f}` °C")
    st.write(f"• **MSE**: `{res_all['mse']:.4f}`")
    st.write(f"• **$R^2$**: `{res_all['r2']:.4f}`")

st.markdown("---
