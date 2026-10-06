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

# 1. 전체 기간 회귀분석 (1908년 기준 경과연수 사용)
df_filtered["경과연수"] = df_filtered["연도"] - 1908
X_all = df_filtered["경과연수"]
Y_all = df_filtered["연평균기온"]

slope_all, intercept_all, r_value_all, p_val_all, std_err_all = stats.linregress(X_all, Y_all)

# 2. 최근 20년 회귀분석
end_year = int(df_filtered["연도"].max())
start_year = int(df_filtered["연도"].min())
recent_start_year = end_year - 19  # 최근 20년간 (예: 2006~2025)

df_recent = df_filtered[df_filtered["연도"] >= recent_start_year]
X_recent = df_recent["경과연수"]
Y_recent = df_recent["연평균기온"]

slope_recent, intercept_recent, r_value_recent, p_val_recent, std_err_recent = stats.linregress(X_recent, Y_recent)

# 100년당 기온 상승 변화량 (기울기 * 100)
rate_100_all = slope_all * 100
rate_100_recent = slope_recent * 100

# ---------------------------------------------------------
# 상단 메인 지표: 100년당 기온 상승 폭 비교
# ---------------------------------------------------------
st.markdown("## 📈 100년당 기온 상승 폭 비교")

col_rate1, col_rate2 = st.columns(2)

with col_rate1:
    st.metric(
        label=f"🌐 전체 기간 기울기 ({start_year}년 ~ {end_year}년)",
        value=f"+{rate_100_all:.2f} °C / 100년",
        help=f"전체 {len(df_filtered)}개 연도의 데이터를 바탕으로 산출한 100년당 기온 상승량입니다."
    )

with col_rate2:
    diff_rate = rate_100_recent - rate_100_all
    st.metric(
        label=f"🔥 최근 20년 기울기 ({recent_start_year}년 ~ {end_year}년)",
        value=f"+{rate_100_recent:.2f} °C / 100년",
        delta=f"전체 평균 대비 {diff_rate:+.2f} °C/100년",
        help="최근 20년 데이터만을 바탕으로 산출한 100년당 기온 상승량입니다."
    )

# 데이터 기본 정보 표시
col1, col2, col3, col4 = st.columns(4)
col1.metric("생성 해 개수", f"{len(df_filtered)}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("전체 상관계수 (r)", f"{r_value_all:.4f}")

st.markdown("---")

# ---------------------------------------------------------
# 슬라이더 및 예측 기온
# ---------------------------------------------------------
selected_year = st.slider(
    "예상 기온을 산출할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 전체 기간 회귀선 기준 예측값 계산
predicted_temp = slope_all * (selected_year - 1908) + intercept_all

st.markdown(f"### 🔮 **{selected_year}년** 서울 예상 평균기온 (전체 기간 추세선 기준)")
st.metric(label=f"{selected_year}년 추정치", value=f"{predicted_temp:.2f} °C")

# ---------------------------------------------------------
# Plotly 시각화 (전체 추세선 및 최근 20년 추세선 함께 표시)
# ---------------------------------------------------------
plot_years = np.arange(1900, 2101)
trendline_all_y = slope_all * (plot_years - 1908) + intercept_all
trendline_recent_y = slope_recent * (plot_years - 1908) + intercept_recent

fig = go.Figure()

# 1. 실제 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_filtered["연도"],
    y=df_filtered["연평균기온"],
    mode="markers",
    name="실제 연평균기온",
    marker=dict(color="#1f77b4", size=7)
))

# 2. 전체 기간 회귀 직선
fig.add_trace(go.Scatter(
    x=plot_years,
    y=trendline_all_y,
    mode="lines",
    name=f"전체 기간 회귀선 (+{rate_100_all:.2f}°C/100년)",
    line=dict(color="#d62728", width=2, dash="dash")
))

# 3. 최근 20년 회귀 직선
fig.add_trace(go.Scatter(
    x=plot_years,
    y=trendline_recent_y,
    mode="lines",
    name=f"최근 20년 회귀선 (+{rate_100_recent:.2f}°C/100년)",
    line=dict(color="#ff7f0e", width=2, dash="dot")
))

# 4. 선택 연도 포인트
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp],
    mode="markers",
    name=f"선택 연도 ({selected_year}년)",
    marker=dict(color="#2ca02c", size=14, symbol="star")
))

fig.update_layout(
    title="서울 연도별 평균기온 추이 및 추세선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
