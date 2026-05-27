import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime
import warnings
warnings.filterwarnings('ignore')

# 設定網頁標題與寬度
st.set_page_config(page_title="AI 泡沫風險監控儀表板", layout="centered")

st.title("📊 AI 泡沫複合風險指數監控")
st.write(f"資料更新時間: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
st.markdown("---")

# 這裡放入您原本的爬蟲與分數計算邏輯 (為了簡潔，這裡以模擬分數示範架構)
# -------------------------------------------------------------
# 假設經過您的模組一到模組四計算後，得出的總分為 68 分
simulated_total_score = 68 
# -------------------------------------------------------------

# 視覺化：風險燈號與分數
if simulated_total_score >= 75:
    st.error(f"🔴 極度危險 (Extreme Bubble) - 分數: {simulated_total_score}/100")
    st.write("估值嚴重脫離基本面，流動性枯竭。建議全面落袋為安，啟動避險策略！")
elif simulated_total_score >= 50:
    st.warning(f"🟡 高度狂熱 (High Hype) - 分數: {simulated_total_score}/100")
    st.write("市場情緒過熱，估值偏高。請密切緊盯供應鏈是否出現砍單或衰退訊號。")
else:
    st.success(f"🟢 健康擴張 (Healthy) - 分數: {simulated_total_score}/100")
    st.write("基本面有強烈支撐，資金流動性充裕，泡沫破裂風險低。")

# 視覺化：進度條 (Gauge)
st.progress(simulated_total_score / 100.0)

st.markdown("### 分項指標風險評分")
col1, col2, col3 = st.columns(3)
col1.metric(label="美國信用利差風險", value="65/100", delta="危險", delta_color="inverse")
col2.metric(label="衍生商品恐慌度", value="30/100", delta="安全", delta_color="normal")
col3.metric(label="科技巨頭估值", value="85/100", delta="極度危險", delta_color="inverse")

st.info("💡 提示：以上數據每天透過 GitHub Actions 自動化抓取最新市場參數進行結算。")
