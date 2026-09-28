# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연도별 기온 예측기")

# 데이터 불러오기 및 전처리
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜를 datetime 변환 후 연도 추출
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

# 회귀분석: 1908년부터 경과한 연수를 독립변수(X)로 설정
df_filtered["경과연수"] = df_filtered["연도"] - 1908
X = df_filtered["경과연수"]
Y = df_filtered["연평균기온"]

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
correlation = r_value

# 메타 정보 계산
num_years = len(df_filtered)
start_year = int(df_filtered["연도"].min())
end_year = int(df_filtered["연도"].max())

# 데이터 요약 메트릭 출력
col1, col2, col3, col4 = st.columns(4)
col1.metric("회귀선 생성에 사용된 해 개수", f"{num_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{correlation:.4f}")

st.markdown("---")

# 연도 선택 슬라이더 (1900년 ~ 2100년)
selected_year = st.slider(
    "예상 기온을 산출할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 선택한 연도의 기온 예측 (1908년 기준 경과연수 적용)
predicted_temp = slope * (selected_year - 1908) + intercept

# 예측 기온 대형 강조 표시
st.markdown(f"### 🔮 **{selected_year}년** 서울 예상 평균기온")
st.metric(label=f"{selected_year}년 추정치", value=f"{predicted_temp:.2f} °C")

# Plotly 그래프 생성
plot_years = np.arange(1900, 2101)
trendline_y = slope * (plot_years - 1908) + intercept

fig = go.Figure()

# 1. 실제 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_filtered["연도"],
    y=df_filtered["연평균기온"],
    mode="markers",
    name="실제 연평균기온",
    marker=dict(color="#1f77b4", size=7)
))

# 2. 회귀 직선
fig.add_trace(go.Scatter(
    x=plot_years,
    y=trendline_y,
    mode="lines",
    name="선형 회귀선",
    line=dict(color="#d62728", width=2, dash="dash")
))

# 3. 선택 연도 포인트
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"선택 연도 ({selected_year}년)",
    marker=dict(color="#2ca02c", size=14, symbol="star")
))

fig.update_layout(
    title="서울 연도별 평균기온 추이 및 회귀선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
