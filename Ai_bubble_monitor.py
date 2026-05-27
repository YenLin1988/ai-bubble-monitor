import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# 頁面設定 (機構級深色模式)
# -------------------------------------------------------------
st.set_page_config(page_title="Global Macro Regime Radar | AI Bubble Monitor", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .reportview-container .main .block-container{ padding-top: 2rem; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #E0E0E0; }
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Global Macro Regime Radar (v4.0 終極加權版)")
st.caption(f"Last Updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC) | Data Source: FRED, Yahoo Finance")
st.markdown("---")

# -------------------------------------------------------------
# 1. 核心參數與資料抓取 (加入 10Y公債 與 美元指數)
# -------------------------------------------------------------
TECH_GIANTS = ['NVDA', 'MSFT', 'META', 'GOOG']
TW_SUPPLY = ['TSM', 'UMC', 'ASX'] 
INFRA_STOCKS = ['MU', 'VRT'] 
COMMODITIES = ['CL=F', 'GC=F'] 

# 預設變數
latest_spread = 3.5; latest_curve = -0.5; vxn = 20.0
avg_ev_ebitda = 30.0; avg_momentum = 0.05; infra_momentum = 0.05; comm_momentum = 0.0
net_liquidity = 6000.0; tnx_yield = 4.2; dxy = 104.0

# [A] 抓取 FRED 數據
try:
    latest_spread = float(pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2', na_values='.').dropna().iloc[-1, 1])
    latest_curve = float(pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y', na_values='.').dropna().iloc[-1, 1])
    walcl = float(pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=WALCL', na_values='.').dropna().iloc[-1, 1]) / 1000
    tga = float(pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=WTREGEN', na_values='.').dropna().iloc[-1, 1])
    rrp = float(pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=RRPONTSYD', na_values='.').dropna().iloc[-1, 1])
    net_liquidity = walcl - tga - rrp 
except: pass

# [B] 抓取 YFinance 數據
def get_20d_momentum(tickers):
    mom_list = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period='20d')['Close']
            mom_list.append((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0])
        except: pass
    return sum(mom_list) / len(mom_list) if mom_list else 0.0

try: vxn = yf.Ticker('^VXN').history(period='5d')['Close'].iloc[-1]
except: pass
try: tnx_yield = yf.Ticker('^TNX').history(period='5d')['Close'].iloc[-1] # 10年期公債殖利率
except: pass
try: dxy = yf.Ticker('DX-Y.NYB').history(period='5d')['Close'].iloc[-1] # 美元指數
except: pass

macro_ev_ebitda = []
for ticker in TECH_GIANTS:
    try: macro_ev_ebitda.append(yf.Ticker(ticker).info.get('enterpriseToEbitda', 30))
    except: pass
avg_ev_ebitda = sum(macro_ev_ebitda) / len(macro_ev_ebitda) if macro_ev_ebitda else 30.0

avg_momentum = get_20d_momentum(TW_SUPPLY)
infra_momentum = get_20d_momentum(INFRA_STOCKS)
comm_momentum = get_20d_momentum(COMMODITIES)

# -------------------------------------------------------------
# 2. 華爾街加權評估系統 (Weighted Scoring Engine)
# -------------------------------------------------------------
def normalize(value, safe_val, danger_val):
    """將單一指標轉化為 0-100 的風險分數"""
    score = ((value - safe_val) / (danger_val - safe_val)) * 100
    return min(100, max(0, score))

# [X軸] AI 基本面動能 (Growth) 
# 台系代工(60%) + 基礎設施(40%)
combined_growth = (avg_momentum * 0.6) + (infra_momentum * 0.4)
growth_score = min(100, max(-100, combined_growth * 1000))

# [Y軸] 總體壓力風險分數轉換 (0-100)
risk_scores = {
    "10Y Treasury Yield (定價引力)": normalize(tnx_yield, 3.5, 5.0),
    "US Dollar Index (全球水閘)": normalize(dxy, 100.0, 107.0),
    "Net Liquidity (絕對資金)": normalize(6200 - net_liquidity, 0, 800), # 低於6.2T開始扣分
    "Yield Curve (衰退警報)": normalize(latest_curve, -0.8, 0.2),
    "Credit Spread (違約風險)": normalize(latest_spread, 3.0, 5.5),
    "Commodities (通膨預期)": normalize(comm_momentum, 0.0, 0.10),
    "EV/EBITDA (估值極限)": normalize(avg_ev_ebitda, 25, 45),
    "VXN Volatility (市場情緒)": normalize(vxn, 15, 35)
}

# 壓力指標比重配置 (Weightings) - 總和 100%
weights = {
    "10Y Treasury Yield (定價引力)": 0.15,  # 15%
    "US Dollar Index (全球水閘)": 0.15,      # 15%
    "Net Liquidity (絕對資金)": 0.15,        # 15%
    "Yield Curve (衰退警報)": 0.15,          # 15%
    "Credit Spread (違約風險)": 0.10,        # 10%
    "EV/EBITDA (估值極限)": 0.10,            # 10%
    "Commodities (通膨預期)": 0.10,          # 10%
    "VXN Volatility (市場情緒)": 0.10        # 10%
}

# 計算加權綜合壓力總分 (Weighted Stress Score)
stress_score = sum(risk_scores[k] * weights[k] for k in risk_scores)

# 判定當前狀態 (Regime)
if growth_score >= 0 and stress_score < 50: current_regime = "🟢 擴張期 (Expansion / Goldilocks)"; regime_color = "#00CC96"
elif growth_score >= 0 and stress_score >= 50: current_regime = "🟡 過熱期 (Overheating / Late Cycle)"; regime_color = "#FFA15A"
elif growth_score < 0 and stress_score >= 50: current_regime = "🔴 泡沫破裂 (Bust / Liquidity Shock)"; regime_color = "#EF553B"
else: current_regime = "🔵 沉澱期 (Contraction / Value Zone)"; regime_color = "#636EFA"

# -------------------------------------------------------------
# 3. 視覺化渲染：散佈圖與 8維度 雷達圖
# -------------------------------------------------------------
st.subheader(f"📍 總體經濟狀態：{current_regime}")
col_left, col_right = st.columns([1.5, 1])

with col_left:
    fig_scatter = go.Figure()
    fig_scatter.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor="rgba(255, 161, 90, 0.1)", line_width=0)
    fig_scatter.add_shape(type="rect", x0=0, y0=-20, x1=100, y1=0, fillcolor="rgba(0, 204, 150, 0.1)", line_width=0)
    fig_scatter.add_shape(type="rect", x0=-100, y0=0, x1=0, y1=100, fillcolor="rgba(239, 85, 59, 0.1)", line_width=0)
    fig_scatter.add_shape(type="rect", x0=-100, y0=-20, x1=0, y1=0, fillcolor="rgba(99, 110, 250, 0.1)", line_width=0)
    fig_scatter.add_trace(go.Scatter(
        x=[growth_score], y=[stress_score], mode='markers+text',
        marker=dict(size=20, color=regime_color, line=dict(width=2, color='white')),
        text=[f"Current<br>Stress: {stress_score:.1f}"], textposition="top center", textfont=dict(color="white")
    ))
    fig_scatter.update_layout(
        title="Macro Regime Map", xaxis_title="AI Growth Momentum (動能)", yaxis_title="Weighted Liquidity Stress (加權總體壓力)",
        xaxis=dict(range=[-100, 100]), yaxis=dict(range=[-20, 100]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=400, margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    # 建立 8 維度雷達圖 (使用真實分數)
    categories = list(risk_scores.keys())
    radar_vals = list(risk_scores.values())
    
    fig_radar = go.Figure(data=go.Scatterpolar(
      r=radar_vals + [radar_vals[0]], theta=categories + [categories[0]],
      fill='toself', fillcolor="rgba(239, 85, 59, 0.4)", line=dict(color="#EF553B")
    ))
    fig_radar.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333"), angularaxis=dict(gridcolor="#333")),
      showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
      title="8-Factor Risk Decomposition", height=400, margin=dict(l=30, r=30, t=40, b=30)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------------------------------------------
# 4. 數據儀表板與加權分析報告
# -------------------------------------------------------------
st.markdown("### 📊 市場真實數據與綜合評估")

# 重新排版為兩排，讓資訊更乾淨
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
row1_col1.metric("10Y Treasury (美債殖利率)", f"{tnx_yield:.2f}%")
row1_col2.metric("US Dollar (美元指數)", f"{dxy:.2f}")
row1_col3.metric("Net Liquidity (淨流動性)", f"${net_liquidity/1000:.2f}T")
row1_col4.metric("10Y-2Y Curve (殖利率曲線)", f"{latest_curve:.2f}%")

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
row2_col1.metric("Credit Spread (信用利差)", f"{latest_spread:.2f}%")
row2_col2.metric("EV/EBITDA (AI估值)", f"{avg_ev_ebitda:.1f}x")
row2_col3.metric("Commodity Mom (通膨預期)", f"{comm_momentum*100:.1f}%")
row2_col4.metric("VXN Index (市場情緒)", f"{vxn:.2f}")

st.markdown("---")
st.markdown("### 📝 加權分析報告與行動建議")

# 找出得分最高的最大風險來源
primary_risk_name = max(risk_scores, key=risk_scores.get)
primary_risk_value = risk_scores[primary_risk_name]

# 顯示各項指標的得分與權重
with st.expander("🔍 點此展開查看【綜合壓力評估矩陣】詳細計算"):
    st.write("本系統採用加權平均法計算總體壓力。單項分數越高代表該領域泡沫或衰退風險越大 (滿分 100)。")
    matrix_df = pd.DataFrame({
        "風險維度": list(risk_scores.keys()),
        "模型風險分數 (0-100)": [f"{v:.1f}" for v in risk_scores.values()],
        "模型配置權重": [f"{w*100:.0f}%" for w in weights.values()]
    })
    st.table(matrix_df)

# 動態文字分析
st.write(f"**📍 核心狀態判讀：** 當前加權壓力總分為 **{stress_score:.1f} / 100**。系統判定目前處於 **{current_regime}**。")

if "擴張期" in current_regime:
    st.info("宏觀經濟數據（美債殖利率、美元指數）均處於溫和區間，且 AI 基礎設施動能強勁。資金環境極其健康，建議維持多頭配置。")
elif "過熱期" in current_regime:
    st.warning("⚠️ **警訊：** 雖然 AI 相關企業基本面仍在支撐大盤，但由美債、美元與流動性組成的「總經地心引力」正在增強。")
    if primary_risk_value > 60:
        st.write(f"**🚨 當前最致命風險：** 系統偵測到 **「{primary_risk_name}」** 已達到危險臨界值。")
        st.write("這意味著市場的估值隨時可能被宏觀數據（如強勢美元引發的資金抽離，或高殖利率引發的殺估值）所吞噬。建議開始對沖風險，買入防禦性資產。")
elif "泡沫破裂" in current_regime:
    st.error("🛑 **全面撤退訊號：** AI 硬體供應鏈已出現疲態，同時美元飆漲、美債殖利率失控或流動性枯竭。資產價格正在進行無差別拋售的重新定價，請嚴格控制現金水位！")
elif "沉澱期" in current_regime:
    st.success("總體經濟的壓力（如美元與美債殖利率）已經見頂回落，雖然 AI 企業的動能還在打底，但估值泡沫已被清洗。這是長線逢低佈局的黃金視窗。")
