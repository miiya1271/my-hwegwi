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
    rate_100 = slope * 100
    return {
        "slope": slope,
        "intercept": intercept,
        "rate_100": rate_100,
        "mae": mae,
        "mse": mse,
        "r2": r2
    }

res_50 = eval_metrics(model_50, y_test, pred_50)
res_100 = eval_metrics(model_100, y_test, pred_100)
res_all = eval_metrics(model_all, y_test, pred_all)

# ---------------------------------------------------------
# 상단 메인 지표: 기울기(100년당 상승률) & 성능 비교 카드
# ---------------------------------------------------------
st.subheader("📊 테스트 데이터(2006~2025년) 대상 예측 성능 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔹 최근 50년 학습 (1956~2005)")
    st.metric("100년당 기온 상승 폭", f"+{res_50['rate_100']:.2f} °C")
    st.caption(f"**학습 데이터**: {len(train_50)}개 연도")
    st.write(f"**회귀식**: $y = {res_50['slope']:.5f}x + ({res_50['intercept']:.3f})$")
    st.write(f"- **MAE** : `{res_50['mae']:.4f}` °C")
    st.write(f"- **MSE** : `{res_50['mse']:.4f}`")
    st.write(f"- **R²**   : `{res_50['r2']:.4f}`")

with col2:
    st.markdown("### 🔸 최근 100년 학습 (1906~2005)")
    delta_slope = res_100['rate_100'] - res_50['rate_100']
    st.metric("100년당 기온 상승 폭", f"+{res_100['rate_100']:.2f} °C", delta=f"{delta_slope:+.2f} °C (vs 50년)")
    st.caption(f"**학습 데이터**: {len(train_100)}개 연도")
    st.write(f"**회귀식**: $y = {res_100['slope']:.5f}x + ({res_100['intercept']:.3f})$")
    st.write(f"- **MAE** : `{res_100['mae']:.4f}` °C")
    st.write(f"- **MSE** : `{res_100['mse']:.4f}`")
    st.write(f"- **R²**   : `{res_100['r2']:.4f}`")

with col3:
    st.markdown("### 🟢 전체 데이터 학습 (전체)")
    st.metric("100년당 기온 상승 폭", f"+{res_all['rate_100']:.2f} °C")
    st.caption(f"**학습 데이터**: {len(train_all)}개 연도")
    st.write(f"**회귀식**: $y = {res_all['slope']:.5f}x + ({res_all['intercept']:.3f})$")
    st.write(f"- **MAE** : `{res_all['mae']:.4f}` °C")
    st.write(f"- **MSE** : `{res_all['mse']:.4f}`")
    st.write(f"- **R²**   : `{res_all['r2']:.4f}`")

st.markdown("---")

# ---------------------------------------------------------
# 표 형태 요약
# ---------------------------------------------------------
summary_table = pd.DataFrame([
    {
        "학습 데이터 기간": "최근 50년 (1956~2005)",
        "학습 샘플 수": f"{len(train_50)}개",
        "기울기 (°C/년)": f"{res_50['slope']:.5f}",
        "100년당 상승 폭": f"+{res_50['rate_100']:.2f} °C",
        "테스트 MAE": f"{res_50['mae']:.4f}",
        "테스트 MSE": f"{res_50['mse']:.4f}",
        "테스트 R²": f"{res_50['r2']:.4f}"
    },
    {
        "학습 데이터 기간": "최근 100년 (1906~2005)",
        "학습 샘플 수": f"{len(train_100)}개",
        "기울기 (°C/년)": f"{res_100['slope']:.5f}",
        "100년당 상승 폭": f"+{res_100['rate_100']:.2f} °C",
        "테스트 MAE": f"{res_100['mae']:.4f}",
        "테스트 MSE": f"{res_100['mse']:.4f}",
        "테스트 R²": f"{res_100['r2']:.4f}"
    },
    {
        "학습 데이터 기간": "전체 기간 (1908~2025)",
        "학습 샘플 수": f"{len(train_all)}개",
        "기울기 (°C/년)": f"{res_all['slope']:.5f}",
        "100년당 상승 폭": f"+{res_all['rate_100']:.2f} °C",
        "테스트 MAE": f"{res_all['mae']:.4f}",
        "테스트 MSE": f"{res_all['mse']:.4f}",
        "테스트 R²": f"{res_all['r2']:.4f}"
    }
])

st.subheader("📋 모델 평가 요약표")
st.dataframe(summary_table, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# 연도 슬라이더 및 시각화
# ---------------------------------------------------------
selected_year = st.slider(
    "예상 기온을 산출할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

pred_val_50 = model_50.predict([[selected_year]])[0]
pred_val_100 = model_100.predict([[selected_year]])[0]
pred_val_all = model_all.predict([[selected_year]])[0]

p_col1, p_col2, p_col3 = st.columns(3)
p_col1.metric(f"50년 학습 모델 ({selected_year}년)", f"{pred_val_50:.2f} °C")
p_col2.metric(f"100년 학습 모델 ({selected_year}년)", f"{pred_val_100:.2f} °C")
p_col3.metric(f"전체 학습 모델 ({selected_year}년)", f"{pred_val_all:.2f} °C")

# Plotly 시각화
plot_years = np.arange(1900, 2101).reshape(-1, 1)
pred_line_50 = model_50.predict(plot_years)
pred_line_100 = model_100.predict(plot_years)
pred_line_all = model_all.predict(plot_years)

fig = go.Figure()

# 1. 과거 훈련 데이터 산점도 (1908~2005)
train_historical = df_filtered[df_filtered["연도"] <= 2005]
fig.add_trace(go.Scatter(
    x=train_historical["연도"],
    y=train_historical["연평균기온"],
    mode="markers",
    name="훈련 데이터 (1908~2005)",
    marker=dict(color="#1f77b4", size=7, opacity=0.7)
))

# 2. 테스트 데이터 산점도 (2006~2025)
fig.add_trace(go.Scatter(
    x=test_data["연도"],
    y=test_data["연평균기온"],
    mode="markers",
    name="테스트 데이터 (2006~2025)",
    marker=dict(color="#ff7f0e", size=9, symbol="diamond")
))

# 3. 모델별 회귀선
fig.add_trace(go.Scatter(
    x=plot_years.flatten(),
    y=pred_line_50,
    mode="lines",
    name=f"최근 50년 학습 회귀선 (+{res_50['rate_100']:.2f}°C/100년)",
    line=dict(color="#e377c2", width=2.5, dash="dash")
))

fig.add_trace(go.Scatter(
    x=plot_years.flatten(),
    y=pred_line_100,
    mode="lines",
    name=f"최근 100년 학습 회귀선 (+{res_100['rate_100']:.2f}°C/100년)",
    line=dict(color="#2ca02c", width=2.5, dash="dot")
))

fig.add_trace(go.Scatter(
    x=plot_years.flatten(),
    y=pred_line_all,
    mode="lines",
    name=f"전체 학습 회귀선 (+{res_all['rate_100']:.2f}°C/100년)",
    line=dict(color="#d62728", width=2)
))

# 4. 선택 연도 포인트
fig.add_trace(go.Scatter(
    x=[selected_year, selected_year, selected_year],
    y=[pred_val_50, pred_val_100, pred_val_all],
    mode="markers",
    name=f"선택 연도 ({selected_year}년) 예측",
    marker=dict(color="black", size=10, symbol="star")
))

fig.update_layout(
    title="서울 연도별 평균기온 및 학습 기간별 회귀선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### 💡 모델 비교 및 시사점
1. **기울기 차이 (온난화 가속화)**:
   - **최근 50년(1956~2005년)** 데이터로 학습한 모델의 기울기가 **최근 100년(1906~2005년)** 모델보다 더 가파릅니다. 이는 20세기 후반 이후 지구 온난화 및 도시화의 영향으로 상승 속도가 점점 빨라졌음을 보여줍니다.
2. **최근 20년(2006~2025년) 예측 성능 비교**:
   - 2006년 이후 실제 기온 상승 추세를 더 잘 반영하는 가파른 기울기의 **최근 50년 학습 모델**이 최근 100년 학습 모델에 비해 오차(**MAE/MSE**)가 더 적고 높은 설명력(**$R^2$**)을 보이는 경향을 나타냅니다.
""")
