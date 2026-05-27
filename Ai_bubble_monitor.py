import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')

# 設定網頁標題與寬度
st.set_page_config(page_title="AI 泡沫風險監控儀表板", layout="centered")

st.title("📊 AI 泡沫複合風險指數監控")
st.write(f"資料更新時間: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
st.markdown("---")

# 1. 設定監控對象
TECH_GIANTS = ['NVDA', 'MSFT', 'META', 'GOOG']
TW_SUPPLY_CHAIN = ['TSM', 'UMC', 'ASX'] 
TODAY = datetime.date.today()

def scale_score(value, min_val, max_val, invert=False):
    if value is None or pd.isna(value): return 50
    score = ((value - min_val) / (max_val - min_val)) * 100
    score = max(0, min(100, score))
    return 100 - score if invert else score

# ================= 核心計算邏輯 =================
scores = {}

# ---- 模組一：總體流動性 (直接爬取 FRED CSV，取代 pandas_datareader) ----
try:
    spread_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2'
    curve_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y2Y'
    
    # 讀取 CSV，並將 '.' (無資料) 轉換為 NaN 後去除
    spread_df = pd.read_csv(spread_url, na_values='.').dropna()
    curve_df = pd.read_csv(curve_url, na_values='.').dropna()
    
    latest_spread = float(spread_df.iloc[-1, 1])
    latest_curve = float(curve_df.iloc[-1, 1])
    
    scores['Credit_Spread'] = scale_score(latest_spread, 3.0, 5.5)
    scores['Yield_Curve'] = scale_score(latest_curve, -0.8, 0.2)
except Exception as e:
    scores['Credit_Spread'], scores['Yield_Curve'] = 50, 50

# ---- 模組二：市場情緒與衍生商品 ----
try:
    vxn = yf.Ticker('^VXN').history(period='5d')['Close'].iloc[-1]
    scores['Market_Panic'] = scale_score(vxn, 15, 35)
except:
    scores['Market_Panic'] = 50

# ---- 模組三：微觀估值與現金流 ----
macro_ev_ebitda = []
fcf_stress = 0
for ticker in TECH_GIANTS:
    try:
        info = yf.Ticker(ticker).info
        macro_ev_ebitda.append(info.get('enterpriseToEbitda', 30))
        if info.get('freeCashflow', 1) < 0: fcf_stress += 25
    except:
        pass
avg_ev_ebitda = sum(macro_ev_ebitda) / len(macro_ev_ebitda) if macro_ev_ebitda else 30
scores['Valuation'] = scale_score(avg_ev_ebitda, 25, 45)
scores['FCF_Warning'] = min(100, fcf_stress)

# ---- 模組四：亞洲供應鏈高頻動能 ----
tw_momentum = []
for adr in TW_SUPPLY_CHAIN:
    try:
        hist = yf.Ticker(adr).history(period='20d')['Close']
        momentum = (hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0]
        tw_momentum.append(momentum)
    except:
        pass
avg_momentum = sum(tw_momentum) / len(tw_momentum) if tw_momentum else 0
scores['Supply_Chain'] = scale_score(avg_momentum * 100, -15, 15, invert=True)

# ---- 計算複合風險總分 ----
total_score = (
    (scores['Credit_Spread'] * 0.15 + scores['Yield_Curve'] * 0.15) + 
    (scores['Market_Panic'] * 0.20) +                                  
    (scores['Valuation'] * 0.20 + scores['FCF_Warning'] * 0.10) +      
    (scores['Supply_Chain'] * 0.20)                                    
)
# ===============================================

# 視覺化：風險燈號與分數
st.subheader("今日綜合評估")
if total_score >= 75:
    st.error(f"🔴 極度危險 (Extreme Bubble) - 總分: {total_score:.1f}/100")
    st.write("估值嚴重脫離基本面，流動性枯竭。建議全面落袋為安，啟動避險策略！")
elif total_score >= 50:
    st.warning(f"🟡 高度狂熱 (High Hype) - 總分: {total_score:.1f}/100")
    st.write("市場情緒過熱，估值偏高。請密切緊盯供應鏈是否出現砍單或衰退訊號。")
else:
    st.success(f"🟢 健康擴張 (Healthy) - 總分: {total_score:.1f}/100")
    st.write("基本面有強烈支撐，資金流動性充裕，泡沫破裂風險低。")

# 視覺化：進度條 (Gauge)
st.progress(total_score / 100.0)

# 視覺化：分項指標
st.markdown("### 📊 分項指標風險評分 (分數越高越危險)")
col1, col2 = st.columns(2)
col1.metric(label="美國信用利差風險", value=f"{scores['Credit_Spread']:.1f}/100")
col2.metric(label="殖利率曲線陡峭化風險", value=f"{scores['Yield_Curve']:.1f}/100")

col3, col4 = st.columns(2)
col3.metric(label="市場恐慌度 (VXN)", value=f"{scores['Market_Panic']:.1f}/100")
col4.metric(label="亞洲硬體供應鏈動能風險", value=f"{scores['Supply_Chain']:.1f}/100")

st.metric(label="AI 巨頭 EV/EBITDA 估值風險", value=f"{scores['Valuation']:.1f}/100", delta=f"當前平均倍數: {avg_ev_ebitda:.1f}x", delta_color="off")

st.info("💡 提示：本模型數據為即時自動抓取最新市場參數進行結算。")
