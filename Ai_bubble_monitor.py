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
st.set_page_config(page_title="Macro Regime Radar | AI Bubble Monitor", layout="wide", initial_sidebar_state="collapsed")

# 自訂 CSS 讓字體與間距更緊湊，模仿彭博終端機風格
st.markdown("""
<style>
    .reportview-container .main .block-container{ padding-top: 2rem; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #E0E0E0; }
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Macro Regime Radar: AI Sector (v2.0)")
st.caption(f"Last Updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC) | Data Source: FRED, Yahoo Finance")
st.markdown("---")

# -------------------------------------------------------------
# 核心資料抓取與處理模組
# -------------------------------------------------------------
TECH_GIANTS = ['NVDA', 'MSFT', 'META', 'GOOG']
TW_SUPPLY_CHAIN = ['TSM', 'UMC', 'ASX'] 
TODAY = datetime.date.today()

# 預設數據 (防止網路抓取失敗)
latest_spread = 3.5; latest_curve = -0.5; vxn = 20.0
avg_ev_ebitda = 30.0; avg_momentum = 0.05; fcf_stress = 0

# (此處為簡化的資料抓取，實際應用可加入歷史資料庫以計算真實 Z-Score)
try:
    spread_df = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2', na_values='.').dropna()
    curve_df = pd.read_csv('https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y', na_values='.').dropna()
    latest_spread = float(spread_df.iloc[-1, 1])
    latest_curve = float(curve_df.iloc[-1, 1])
except: pass

try:
    vxn = yf.Ticker('^VXN').history(period='5d')['Close'].iloc[-1]
except: pass

macro_ev_ebitda = []
for ticker in TECH_GIANTS:
    try:
        info = yf.Ticker(ticker).info
        macro_ev_ebitda.append(info.get('enterpriseToEbitda', 30))
        if info.get('freeCashflow', 1) < 0: fcf_stress += 1
    except: pass
avg_ev_ebitda = sum(macro_ev_ebitda) / len(macro_ev_ebitda) if macro_ev_ebitda else 30.0

tw_momentum = []
for adr in TW_SUPPLY_CHAIN:
    try:
        hist = yf.Ticker(adr).history(period='20d')['Close']
        tw_momentum.append((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0])
    except: pass
avg_momentum = sum(tw_momentum) / len(tw_momentum) if tw_momentum else 0.05

# -------------------------------------------------------------
# 華爾街核心邏輯：狀態辨識 (Regime Identification)
# 將 5 個指標壓縮為 2 個維度：X軸(動能)、Y軸(壓力)
# -------------------------------------------------------------
# 1. 經濟與 AI 動能 (Growth Momentum): 越高越好 (-100 to 100)
# 取決於：硬體供應鏈動能 (正向) + 自由現金流健康度 (負向)
growth_score = min(100, max(-100, (avg_momentum * 1000) - (fcf_stress * 25)))

# 2. 流動性與估值壓力 (Liquidity & Valuation Stress): 越高越危險 (-100 to 100)
# 取決於：信用利差 (大於4%飆升) + 殖利率曲線陡峭化 (大於0飆升) + EV/EBITDA估值 (大於35飆升)
stress_score = 0
if latest_spread > 4.0: stress_score += (latest_spread - 4.0) * 20
if latest_curve > 0: stress_score += latest_curve * 50
if avg_ev_ebitda > 35: stress_score += (avg_ev_ebitda - 35) * 5
stress_score = min(100, stress_score)

# 判定當前狀態 (Regime)
current_regime = "未知"
regime_color = "gray"
if growth_score >= 0 and stress_score < 50:
    current_regime = "🟢 擴張期 (Expansion / Goldilocks)"
    regime_color = "#00CC96"
elif growth_score >= 0 and stress_score >= 50:
    current_regime = "🟡 過熱期 (Overheating / Late Cycle)"
    regime_color = "#FFA15A"
elif growth_score < 0 and stress_score >= 50:
    current_regime = "🔴 泡沫破裂 (Bust / Liquidity Shock)"
    regime_color = "#EF553B"
elif growth_score < 0 and stress_score < 50:
    current_regime = "🔵 沉澱期 (Contraction / Value Zone)"
    regime_color = "#636EFA"

# -------------------------------------------------------------
# 儀表板 UI 渲染
# -------------------------------------------------------------
# 頂部狀態列
st.subheader(f"Current Macro Regime: {current_regime}")
st.write("機構級四象限分析：X軸為 AI 基本面動能，Y軸為總經流動性壓力。")

col_left, col_right = st.columns([1.5, 1])

# 左側：四象限散佈圖 (Regime Map)
with col_left:
    fig_scatter = go.Figure()
    
    # 畫四個象限的背景色塊 (輔助視覺)
    fig_scatter.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, fillcolor="rgba(255, 161, 90, 0.1)", line_width=0) # 右上: 過熱
    fig_scatter.add_shape(type="rect", x0=0, y0=-20, x1=100, y1=0, fillcolor="rgba(0, 204, 150, 0.1)", line_width=0) # 右下: 擴張
    fig_scatter.add_shape(type="rect", x0=-100, y0=0, x1=0, y1=100, fillcolor="rgba(239, 85, 59, 0.1)", line_width=0) # 左上: 破裂
    fig_scatter.add_shape(type="rect", x0=-100, y0=-20, x1=0, y1=0, fillcolor="rgba(99, 110, 250, 0.1)", line_width=0) # 左下: 沉澱

    # 標示當前位置點
    fig_scatter.add_trace(go.Scatter(
        x=[growth_score], y=[stress_score],
        mode='markers+text',
        marker=dict(size=20, color=regime_color, line=dict(width=2, color='white')),
        text=["Current State"],
        textposition="top center",
        textfont=dict(color="white")
    ))
    
    fig_scatter.update_layout(
        title="AI Sector Regime Map",
        xaxis_title="Growth Momentum (動能)",
        yaxis_title="Liquidity Stress (壓力)",
        xaxis=dict(range=[-100, 100], zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        yaxis=dict(range=[-20, 100], zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# 右側：雷達圖 (Risk Decomposition)
with col_right:
    # 正規化各指標以繪製雷達圖 (0-100)
    radar_vals = [
        min(100, max(0, (latest_spread - 2) * 25)),        # Credit
        min(100, max(0, (latest_curve + 1) * 50)),         # Curve
        min(100, max(0, (vxn - 10) * 3)),                  # Volatility
        min(100, max(0, (avg_ev_ebitda - 15) * 3)),        # Valuation
        min(100, max(0, (-avg_momentum) * 500))            # Supply Chain (負動能轉風險)
    ]
    categories = ['Credit Spread', 'Yield Curve', 'Volatility (VXN)', 'EV/EBITDA', 'Supply Chain Weakness']
    
    fig_radar = go.Figure(data=go.Scatterpolar(
      r=radar_vals + [radar_vals[0]], # 閉合雷達圖
      theta=categories + [categories[0]],
      fill='toself',
      fillcolor="rgba(239, 85, 59, 0.4)",
      line=dict(color="#EF553B")
    ))
    
    fig_radar.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, 100], gridcolor="#333"),
        angularaxis=dict(gridcolor="#333")
      ),
      showlegend=False,
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font=dict(color="white"),
      title="Risk Decomposition (風險拆解)",
      height=400,
      margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------------------------------------------
# 底部：原始數據 Raw Data 儀表板
# -------------------------------------------------------------
st.markdown("### 📊 Raw Macro Data")
met_col1, met_col2, met_col3, met_col4, met_col5 = st.columns(5)

met_col1.metric("Credit Spread", f"{latest_spread:.2f}%")
met_col2.metric("10Y-2Y Curve", f"{latest_curve:.2f}%")
met_col3.metric("VXN Index", f"{vxn:.2f}")
met_col4.metric("EV/EBITDA Avg", f"{avg_ev_ebitda:.1f}x")
met_col5.metric("Supply Chain (20d)", f"{avg_momentum*100:.2f}%")
# -------------------------------------------------------------
# 動態分析與風險總結 (Dynamic Analysis & Summary)
# -------------------------------------------------------------
st.markdown("---")
st.markdown("### 📝 市場狀態解析與行動建議")

# 1. 尋找最大風險來源
risk_factors = {
    "信用利差 (Credit Spread)": radar_vals[0],
    "殖利率曲線 (Yield Curve)": radar_vals[1],
    "市場恐慌度 (Volatility)": radar_vals[2],
    "估值倍數 (EV/EBITDA)": radar_vals[3],
    "供應鏈衰退 (Supply Chain)": radar_vals[4]
}
# 找出雷達圖中數值最大的那一項
primary_risk_name = max(risk_factors, key=risk_factors.get)
primary_risk_value = risk_factors[primary_risk_name]

# 2. 組合分析報告文字
st.write(f"**📍 核心狀態判讀：** 系統判定目前處於 **{current_regime}**。")

if "擴張期" in current_regime:
    st.info("目前 AI 基本面動能強勁（如台系硬體供應鏈持續創高），且總體流動性壓力極低。這是一個『金髮女孩』的完美環境，建議維持既有多頭部位，享受市場紅利。")
elif "過熱期" in current_regime:
    st.warning(f"請注意！雖然 AI 基本面依然強勁，支撐著股市動能，但總經環境已亮起紅燈。目前的壓力主要來自於總體經濟面的緊縮。")
    if primary_risk_value > 60:
        st.write(f"**🚨 首要風險來源：** 目前雷達圖顯示最大的風險來自於 **「{primary_risk_name}」**。")
        st.write("這意味著市場正處於『基本面與資金面拔河』的階段。建議不要過度追高，可考慮將部分獲利了結，或買入賣權 (Put) 替部位買保險。")
elif "泡沫破裂" in current_regime:
    st.error("危險訊號！AI 硬體供應鏈已出現疲態（動能轉負），且總體流動性正在枯竭。基本面與資金面遭到雙殺，市場即將或正在經歷劇烈修正。")
    st.write("🛑 **行動建議：** 建議大幅降低科技股曝險，提高現金比例，並轉進防禦型資產（如短天期美債）。")
elif "沉澱期" in current_regime:
    st.success("目前市場處於低動能且低壓力的狀態。這通常發生在泡沫破裂後的打底期。")
    st.write("💡 **行動建議：** 恐慌情緒已散去，估值回到合理區間，是長線投資人開始分批佈局優質 AI 龍頭股的絕佳時機。")

# 免責聲明
st.caption("Disclaimer: 本模型由量化指標自動生成，僅供觀察總經環境與市場情緒參考，不構成任何具體投資建議。")
