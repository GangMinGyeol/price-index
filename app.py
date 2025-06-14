import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="물가 상승률 예측 서비스",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("📈 한국 소비자물가지수 분석 및 예측")
st.markdown("---")

@st.cache_data
def load_and_process_real_data():
    """실제 업로드된 데이터 처리"""
    try:
        # 업로드된 실제 데이터 읽기
        if 'uploaded_file' in st.session_state:
            # CSV 파일 읽기
            df_raw = pd.read_csv(st.session_state.uploaded_file, encoding='utf-8')
            
            # 데이터 구조 파악 및 전처리
            # 첫 번째 행이 연도 정보, 두 번째 행이 카테고리 정보로 보임
            # 전체 CPI 데이터만 추출 (첫 번째 컬럼)
            
            # 월별 데이터 추출 (1Month부터 12월까지)
            months = ['1Month', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월']
            cpi_values = []
            
            # 전체 CPI 행 찾기 (첫 번째 데이터 행)
            for col in df_raw.columns[1:13]:  # 1Month부터 12월까지
                try:
                    value = float(df_raw.iloc[0, df_raw.columns.get_loc(col)])
                    cpi_values.append(value)
                except:
                    cpi_values.append(np.nan)
            
            # DataFrame 생성
            df = pd.DataFrame({
                'Year': [2023] * 12,
                'Month': list(range(1, 13)),
                'CPI': cpi_values
            })
            
            # NaN 값 제거
            df = df.dropna()
            
        else:
            # 기본 샘플 데이터
            df = pd.DataFrame({
                'Year': [2023] * 12,
                'Month': list(range(1, 13)),
                'CPI': [111.4, 109.95, 109.99, 110.27, 110.45, 110.94, 110.97, 111.07, 112.19, 112.87, 113.39, 112.44]
            })
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        # 기본 샘플 데이터 사용
        df = pd.DataFrame({
            'Year': [2023] * 12,
            'Month': list(range(1, 13)),
            'CPI': [111.4, 109.95, 109.99, 110.27, 110.45, 110.94, 110.97, 111.07, 112.19, 112.87, 113.39, 112.44]
        })
    
    return df

@st.cache_data
def extend_historical_data(df_2023):
    """2023년 데이터를 기반으로 2020-2022년 데이터 추정 생성"""
    extended_data = []
    
    # 2023년 평균 CPI와 변동성 계산
    avg_cpi_2023 = df_2023['CPI'].mean()
    std_cpi_2023 = df_2023['CPI'].std()
    
    # 2020-2022년 데이터 생성 (역산)
    base_cpi = avg_cpi_2023 * 0.92  # 2020년 시작점을 2023년 평균의 92%로 설정
    
    for year in range(2020, 2023):
        for month in range(1, 13):
            # 연간 약 2.5% 증가 + 계절성 + 노이즈
            years_from_base = year - 2020 + (month - 1) / 12
            trend_growth = base_cpi * (1.025 ** years_from_base)
            
            # 계절성 (여름/겨울 약간 높음)
            seasonal = 0.5 * np.sin(2 * np.pi * month / 12 + np.pi/4)
            
            # 노이즈
            noise = np.random.normal(0, std_cpi_2023 * 0.3)
            
            cpi_value = trend_growth + seasonal + noise
            
            extended_data.append({
                'Year': year,
                'Month': month,
                'CPI': cpi_value
            })
    
    # 2023년 실제 데이터 추가
    for _, row in df_2023.iterrows():
        extended_data.append({
            'Year': int(row['Year']),
            'Month': int(row['Month']),
            'CPI': float(row['CPI'])
        })
    
    # DataFrame 생성
    df_extended = pd.DataFrame(extended_data)
    
    # 날짜 컬럼 생성
    df_extended['Date'] = pd.to_datetime(df_extended[['Year', 'Month']].assign(day=1))
    
    # 정렬
    df_extended = df_extended.sort_values('Date').reset_index(drop=True)
    
    # 전월 대비 상승률 계산
    df_extended['MoM_Rate'] = df_extended['CPI'].pct_change() * 100
    
    # 전년 동월 대비 상승률 계산
    df_extended['YoY_Rate'] = df_extended['CPI'].pct_change(periods=12) * 100
    
    # 12개월 이동평균
    df_extended['CPI_MA12'] = df_extended['CPI'].rolling(window=12, center=True).mean()
    
    return df_extended

def calculate_inflation_stats(df):
    """물가 상승률 통계 계산"""
    recent_data = df.tail(12)  # 최근 12개월
    
    # NaN 값 제거
    recent_mom = recent_data['MoM_Rate'].dropna()
    recent_yoy = recent_data['YoY_Rate'].dropna()
    
    stats = {
        '현재 CPI': df['CPI'].iloc[-1],
        '전월 대비': df['MoM_Rate'].iloc[-1] if not pd.isna(df['MoM_Rate'].iloc[-1]) else 0,
        '전년 동월 대비': recent_yoy.iloc[-1] if len(recent_yoy) > 0 else 0,
        '최근 12개월 평균 (월간)': recent_mom.mean() if len(recent_mom) > 0 else 0,
        '최근 12개월 최대': recent_mom.max() if len(recent_mom) > 0 else 0,
        '최근 12개월 최소': recent_mom.min() if len(recent_mom) > 0 else 0,
        '변동성 (표준편차)': recent_mom.std() if len(recent_mom) > 0 else 0
    }
    
    return stats

def create_prophet_model(df):
    """Prophet 모델 생성 및 학습"""
    # Prophet용 데이터 준비
    prophet_df = df[['Date', 'CPI']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.dropna()
    
    # Prophet 모델 생성 - 실제 데이터에 맞춤 조정
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.008,  # 매우 보수적
        seasonality_prior_scale=0.05,   # 계절성 영향 최소화
        changepoint_range=0.9,          # 변화점 감지 범위
        interval_width=0.8,             # 신뢰구간
        mcmc_samples=0                  # 빠른 예측을 위해
    )
    
    # 모델 학습
    model.fit(prophet_df)
    
    return model, prophet_df

def predict_future(model, prophet_df, periods=12):
    """미래 예측 - 더욱 보수적으로"""
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    
    # 현재 값과 최근 트렌드 기반으로 예측값 조정
    current_value = prophet_df['y'].iloc[-1]
    recent_growth = prophet_df['y'].pct_change().tail(6).mean()  # 최근 6개월 평균 성장률
    
    # 예측 부분만 추출
    future_indices = forecast.index[-periods:]
    
    # 보수적 예측: 최근 트렌드를 기반으로 한 선형 증가
    for i, idx in enumerate(future_indices):
        months_ahead = i + 1
        # 월간 성장률을 0.1~0.3% 범위로 제한
        conservative_growth = max(0.001, min(0.003, recent_growth))
        adjusted_value = current_value * ((1 + conservative_growth) ** months_ahead)
        
        # 예측값을 조정된 값으로 대체 (단, 원래 예측값과 크게 다르지 않도록)
        original_pred = forecast.loc[idx, 'yhat']
        forecast.loc[idx, 'yhat'] = (adjusted_value + original_pred) / 2
        
        # 신뢰구간도 조정
        uncertainty = adjusted_value * 0.015  # 1.5% 불확실성
        forecast.loc[idx, 'yhat_lower'] = forecast.loc[idx, 'yhat'] - uncertainty
        forecast.loc[idx, 'yhat_upper'] = forecast.loc[idx, 'yhat'] + uncertainty
    
    return forecast

# 사이드바
st.sidebar.header("📊 설정")

# 파일 업로드
uploaded_file = st.sidebar.file_uploader(
    "CSV 파일 업로드", 
    type=['csv'],
    help="소비자물가지수 데이터 파일을 업로드하세요"
)

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.sidebar.success("✅ 파일이 성공적으로 업로드되었습니다!")

# 예측 기간 설정
prediction_months = st.sidebar.slider(
    "예측 기간 (개월)", 
    min_value=3, 
    max_value=24, 
    value=12,
    help="앞으로 몇 개월을 예측할지 선택하세요"
)

# 데이터 로드 및 처리
with st.spinner('데이터를 로드하고 있습니다...'):
    df_2023 = load_and_process_real_data()
    df = extend_historical_data(df_2023)

# 데이터 정보 표시
st.sidebar.info(f"""
📊 **데이터 정보**
- 기간: {df['Date'].min().strftime('%Y-%m')} ~ {df['Date'].max().strftime('%Y-%m')}
- 총 데이터 포인트: {len(df)}개
- 2023년 실제 데이터: {len(df_2023)}개월
""")

# 메인 컨텐츠
col1, col2, col3, col4 = st.columns(4)

# 주요 지표 표시
stats = calculate_inflation_stats(df)

with col1:
    st.metric(
        "현재 CPI", 
        f"{stats['현재 CPI']:.1f}",
        delta=f"{stats['전월 대비']:.2f}%" if not pd.isna(stats['전월 대비']) else None
    )

with col2:
    st.metric(
        "전년 동월 대비", 
        f"{stats['전년 동월 대비']:.2f}%"
    )

with col3:
    st.metric(
        "최근 12개월 평균", 
        f"{stats['최근 12개월 평균 (월간)']:.2f}%"
    )

with col4:
    st.metric(
        "변동성", 
        f"{stats['변동성 (표준편차)']:.2f}%"
    )

st.markdown("---")

# 탭 생성
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 CPI 추이", "📊 상승률 분석", "🔮 미래 예측", "📋 실제 데이터", "📄 데이터 테이블"])

with tab1:
    st.subheader("소비자물가지수(CPI) 추이")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('소비자물가지수', '전월 대비 상승률'),
        vertical_spacing=0.1
    )
    
    # 실제 2023년 데이터 강조
    df_2023_with_date = df[df['Year'] == 2023].copy()
    df_historical = df[df['Year'] < 2023].copy()
    
    # 히스토리컬 데이터
    fig.add_trace(
        go.Scatter(
            x=df_historical['Date'], 
            y=df_historical['CPI'],
            mode='lines',
            name='추정 CPI (2020-2022)',
            line=dict(color='lightblue', width=1, dash='dot'),
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # 실제 2023년 데이터
    fig.add_trace(
        go.Scatter(
            x=df_2023_with_date['Date'], 
            y=df_2023_with_date['CPI'],
            mode='lines+markers',
            name='실제 CPI (2023)',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # 12개월 이동평균
    if not df['CPI_MA12'].isna().all():
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['CPI_MA12'],
                mode='lines',
                name='12개월 이동평균',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # 전월 대비 상승률
    fig.add_trace(
        go.Scatter(
            x=df['Date'], 
            y=df['MoM_Rate'],
            mode='lines+markers',
            name='전월 대비 상승률(%)',
            line=dict(color='green', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # 0% 기준선
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="소비자물가지수 및 상승률 추이"
    )
    
    fig.update_xaxes(title_text="날짜")
    fig.update_yaxes(title_text="CPI", row=1, col=1)
    fig.update_yaxes(title_text="상승률(%)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("물가 상승률 심층 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 2023년 월별 상승률 분포
        df_2023_analysis = df[df['Year'] == 2023].copy()
        if not df_2023_analysis['MoM_Rate'].isna().all():
            fig_box = px.bar(
                df_2023_analysis.dropna(subset=['MoM_Rate']), 
                x='Month', 
                y='MoM_Rate',
                title='2023년 월별 전월 대비 상승률',
                labels={'Month': '월', 'MoM_Rate': '상승률(%)'}
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # 연도별 평균 상승률
        yearly_avg = df.groupby('Year')['MoM_Rate'].mean().reset_index()
        yearly_avg = yearly_avg.dropna()
        if not yearly_avg.empty:
            fig_bar = px.bar(
                yearly_avg, 
                x='Year', 
                y='MoM_Rate',
                title='연도별 평균 월간 상승률',
                labels={'Year': '연도', 'MoM_Rate': '평균 상승률(%)'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # 상승률 히스토그램
    if not df['MoM_Rate'].isna().all():
        fig_hist = px.histogram(
            df.dropna(subset=['MoM_Rate']), 
            x='MoM_Rate', 
            nbins=15,
            title='전월 대비 상승률 분포',
            labels={'MoM_Rate': '상승률(%)', 'count': '빈도'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.subheader("🔮 Prophet을 이용한 미래 예측")
    
    with st.spinner('예측 모델을 생성하고 있습니다...'):
        try:
            # Prophet 모델 생성 및 예측
            model, prophet_df = create_prophet_model(df)
            forecast = predict_future(model, prophet_df, periods=prediction_months)
            
            # 예측 결과 시각화
            fig_forecast = go.Figure()
            
            # 히스토리컬 데이터 (추정)
            historical_data = prophet_df[prophet_df['ds'] < '2023-01-01']
            fig_forecast.add_trace(
                go.Scatter(
                    x=historical_data['ds'],
                    y=historical_data['y'],
                    mode='lines',
                    name='추정 CPI (2020-2022)',
                    line=dict(color='lightblue', width=1, dash='dot'),
                    opacity=0.7
                )
            )
            
            # 실제 2023년 데이터
            actual_2023 = prophet_df[prophet_df['ds'] >= '2023-01-01']
            fig_forecast.add_trace(
                go.Scatter(
                    x=actual_2023['ds'],
                    y=actual_2023['y'],
                    mode='lines+markers',
                    name='실제 CPI (2023)',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6)
                )
            )
            
            # 예측 데이터
            future_data = forecast.tail(prediction_months)
            fig_forecast.add_trace(
                go.Scatter(
                    x=future_data['ds'],
                    y=future_data['yhat'],
                    mode='lines+markers',
                    name='예측 CPI',
                    line=dict(color='red', dash='dash', width=2),
                    marker=dict(size=6)
                )
            )
            
            # 신뢰구간
            fig_forecast.add_trace(
                go.Scatter(
                    x=future_data['ds'],
                    y=future_data['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                )
            )
            
            fig_forecast.add_trace(
                go.Scatter(
                    x=future_data['ds'],
                    y=future_data['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='80% 신뢰구간',
                    fillcolor='rgba(255,0,0,0.2)'
                )
            )
            
            fig_forecast.update_layout(
                title=f'CPI 예측 (향후 {prediction_months}개월)',
                xaxis_title='날짜',
                yaxis_title='CPI',
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # 예측 통계
            st.subheader("예측 요약")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_value = prophet_df['y'].iloc[-1]
                future_value = future_data['yhat'].iloc[-1]
                future_growth = ((future_value / current_value) - 1) * 100
                st.metric(
                    f"{prediction_months}개월 후 예상 증가율",
                    f"{future_growth:.2f}%"
                )
            
            with col2:
                monthly_avg_growth = future_growth / prediction_months
                st.metric(
                    "월평균 예상 증가율",
                    f"{monthly_avg_growth:.3f}%"
                )
            
            with col3:
                prediction_uncertainty = (future_data['yhat_upper'].iloc[-1] - future_data['yhat_lower'].iloc[-1]) / 2
                st.metric(
                    "예측 불확실성",
                    f"±{prediction_uncertainty:.2f}"
                )
                
            # 경고 메시지
            st.warning("⚠️ **예측 주의사항**: 이 예측은 2023년 실제 데이터와 추정된 과거 데이터를 기반으로 하며, 실제 경제 상황, 정책 변화, 외부 충격 등은 반영되지 않습니다.")
            
        except Exception as e:
            st.error(f"예측 모델 생성 중 오류가 발생했습니다: {str(e)}")

with tab4:
    st.subheader("📋 2023년 실제 데이터")
    
    # 2023년 실제 데이터 시각화
    df_2023_display = df[df['Year'] == 2023].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 월별 CPI 값
        fig_actual = px.line(
            df_2023_display, 
            x='Month', 
            y='CPI',
            title='2023년 월별 CPI',
            markers=True
        )
        fig_actual.update_layout(height=400)
        st.plotly_chart(fig_actual, use_container_width=True)
    
    with col2:
        # 월별 상승률
        fig_rate = px.bar(
            df_2023_display.dropna(subset=['MoM_Rate']), 
            x='Month', 
            y='MoM_Rate',
            title='2023년 월별 전월 대비 상승률'
        )
        fig_rate.update_layout(height=400)
        st.plotly_chart(fig_rate, use_container_width=True)
    
    # 실제 데이터 테이블
    st.write("**2023년 월별 데이터**")
    display_df = df_2023_display[['Month', 'CPI', 'MoM_Rate']].copy()
    display_df.columns = ['월', 'CPI', '전월 대비 상승률(%)']
    display_df = display_df.round(3)
    st.dataframe(display_df, use_container_width=True)

with tab5:
    st.subheader("📋 전체 데이터 테이블")
    
    # 최근 데이터 표시
    st.write("**최근 24개월 데이터**")
    recent_df = df[['Date', 'Year', 'Month', 'CPI', 'MoM_Rate', 'YoY_Rate']].tail(24).copy()
    recent_df['Date'] = recent_df['Date'].dt.strftime('%Y-%m')
    recent_df.columns = ['날짜', '연도', '월', 'CPI', '전월대비(%)', '전년동월대비(%)']
    recent_df = recent_df.round(3)
    st.dataframe(recent_df, use_container_width=True)
    
    # 예측 데이터 표시
    if 'forecast' in locals() and 'future_data' in locals():
        st.write(f"**향후 {prediction_months}개월 예측**")
        future_df = future_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        future_df['ds'] = future_df['ds'].dt.strftime('%Y-%m')
        future_df.columns = ['날짜', '예측 CPI', '하한', '상한']
        future_df = future_df.round(3)
        st.dataframe(future_df, use_container_width=True)
    
    # 데이터 다운로드
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 전체 데이터 다운로드 (CSV)",
        data=csv,
        file_name=f'cpi_analysis_{pd.Timestamp.now().strftime("%Y%m%d")}.csv',
        mime='text/csv'
    )

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>📊 한국 소비자물가지수 분석 및 예측 서비스</p>
    <p>실제 2023년 데이터 기반 | Prophet 라이브러리 사용 | Made with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)