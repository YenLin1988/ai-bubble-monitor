import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# 頁面設定
# =============================================================
st.set_page_config(
    page_title="Global Macro Regime Radar | AI Bubble Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================
# 機構級深色主題 CSS
# =============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

.stApp {
    background-color: #0D1117 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #E6EDF3;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem !important;
    max-width: 1440px !important;
}

/* Regime Banner */
.regime-banner {
    background: linear-gradient(135deg, #161B22 0%, #1C2333 100%);
    border: 1px solid #21262D;
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.regime-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.regime-banner.safe::before    { background: linear-gradient(90deg, #2EA043, #39D2C0); }
.regime-banner.caution::before { background: linear-gradient(90deg, #D29922, #E3B341); }
.regime-banner.danger::before  { background: linear-gradient(90deg, #F85149, #FF4757); }
.regime-banner.cool::before    { background: linear-gradient(90deg, #58A6FF, #BC8CFF); }

.regime-flex {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}
.regime-left { display: flex; align-items: center; gap: 16px; }
.regime-dot {
    width: 14px; height: 14px;
    border-radius: 50%;
    box-shadow: 0 0 12px currentColor;
}
.regime-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
    color: #E6EDF3;
}
.regime-sub {
    font-size: 0.8rem;
    color: #8B949E;
    margin-top: 2px;
}
.regime-right { text-align: right; }
.regime-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1;
}
.regime-label {
    font-size: 0.7rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* KPI Cards */
div[data-testid="stMetric"] {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
    padding: 18px 20px !important;
    transition: border-color 0.2s ease, transform 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #30363D !important;
    transform: translateY(-1px);
}
div[data-testid="stMetric"] label {
    color: #8B949E !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #E6EDF3 !important;
}

/* Section Headers */
h1 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #E6EDF3 !important;
    letter-spacing: -0.03em !important;
    border: none !important;
}
h2 {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #8B949E !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #E6EDF3 !important;
}

/* Chart Containers */
.stPlotlyChart {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
    padding: 4px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #161B22 !important;
    border: 1px solid #21262D !important;
    border-radius: 10px !important;
    color: #8B949E !important;
    font-size: 0.85rem !important;
}
.streamlit-expanderContent {
    background-color: #161B22 !important;
    border: 1px solid #21262D !important;
    border-top: none !important;
}

/* Tables */
.stTable, .dataframe { background-color: #161B22 !important; }
.stTable th {
    background-color: #1C2333 !important;
    color: #8B949E !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
}
.stTable td {
    color: #E6EDF3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Alerts */
.stAlert > div { border-radius: 10px !important; font-size: 0.9rem !important; line-height: 1.6 !important; }

/* Dividers */
hr { border-color: #21262D !important; margin: 24px 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #30363D; border-radius: 3px; }

/* Risk Badge */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}
.risk-badge.safe    { background: rgba(46,160,67,0.15);  color: #2EA043; border: 1px solid rgba(46,160,67,0.3); }
.risk-badge.caution { background: rgba(210,153,34,0.15); color: #D29922; border: 1px solid rgba(210,153,34,0.3); }
.risk-badge.danger  { background: rgba(248,81,73,0.15);  color: #F85149; border: 1px solid rgba(248,81,73,0.3); }
.risk-badge.cool    { background: rgba(88,166,255,0.15); color: #58A6FF; border: 1px solid rgba(88,166,255,0.3); }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: #161B22; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 8px;
    color: #8B949E;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 8px 16px;
}
.stTabs [aria-selected="true"] { background-color: #21262D !important; color: #E6EDF3 !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================
# Plotly 主題
# =============================================================
PLOTLY_BASE = dict(
    plot_bgcolor="#161B22",
    paper_bgcolor="#161B22",
    font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#8B949E"),
    margin=dict(l=48, r=24, t=48, b=40),
    hoverlabel=dict(bgcolor="#1C2333", bordercolor="#30363D",
                    font=dict(family="Inter", size=13, color="#E6EDF3")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262D", borderwidth=1,
                font=dict(size=11, color="#8B949E"))
)

PLOTLY_AXIS = dict(
    xaxis=dict(gridcolor="#21262D", zerolinecolor="#30363D",
               tickfont=dict(family="JetBrains Mono", size=11, color="#484F58"),
               title_font=dict(size=12, color="#8B949E")),
    yaxis=dict(gridcolor="#21262D", zerolinecolor="#30363D",
               tickfont=dict(family="JetBrains Mono", size=11, color="#484F58"),
               title_font=dict(size=12, color="#8B949E")),
)

TITLE_FONT = dict(size=14, color="#E6EDF3", family="Inter")

def apply_theme(fig, title_text=None, **kwargs):
    layout = {**PLOTLY_BASE, **PLOTLY_AXIS, **kwargs}
    if title_text:
        layout['title'] = dict(text=title_text, font=TITLE_FONT, x=0.02, xanchor="left")
    fig.update_layout(**layout)
    return fig

CHART_COLORS = ["#58A6FF", "#BC8CFF", "#39D2C0", "#F0883E", "#F778BA"]

def risk_color(score):
    if score < 25: return "#2EA043"
    elif score < 50: return "#39D2C0"
    elif score < 65: return "#D29922"
    elif score < 80: return "#E3B341"
    else: return "#F85149"

# =============================================================
# 1. 核心資料抓取
# =============================================================
TECH_GIANTS = ['NVDA', 'MSFT', 'META', 'GOOG', 'AMZN']
TW_SUPPLY = ['TSM', 'UMC', 'ASX']
INFRA_STOCKS = ['MU', 'VRT']
COMMODITIES = ['CL=F', 'GC=F']
HYPERSCALERS = ['MSFT', 'GOOG', 'AMZN', 'META']

latest_spread = 3.5; latest_curve = -0.5; vxn = 20.0
avg_ev_ebitda = 30.0; avg_momentum = 0.05; infra_momentum = 0.05; comm_momentum = 0.0
net_liquidity = 6000.0; tnx_yield = 4.2; dxy = 104.0
margin_debt = None; put_call_ratio = None
walcl = 0; tga = 0; rrp = 0
fed_funds = 5.25; m2_yoy = 0.0; nfci = 0.0; unemployment = 3.7; consumer_sentiment = 65.0
sox_momentum = 0.0; btc_momentum = 0.0

# [A] FRED 數據
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id):
    try:
        df = pd.read_csv(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}', na_values='.')
        df = df.dropna()
        return df
    except:
        return None

fred_data = {}
fred_ids = {
    'spread': 'BAMLH0A0HYM2',
    'curve': 'T10Y2Y',
    'walcl': 'WALCL',
    'tga': 'WTREGEN',
    'rrp': 'RRPONTSYD',
    'margin_debt': 'BOGZ1FL663067003Q',
    'wilshire': 'WILL5000PR',
    'gdp': 'GDP',
    'fedfunds': 'FEDFUNDS',
    'm2': 'M2SL',
    'nfci': 'NFCI',
    'unrate': 'UNRATE',
    'umcsent': 'UMCSENT',
}

for key, sid in fred_ids.items():
    fred_data[key] = fetch_fred_series(sid)

try:
    latest_spread = float(fred_data['spread'].iloc[-1, 1])
except: pass
try:
    latest_curve = float(fred_data['curve'].iloc[-1, 1])
except: pass
try:
    walcl = float(fred_data['walcl'].iloc[-1, 1]) / 1000
    tga = float(fred_data['tga'].iloc[-1, 1]) / 1000
    rrp = float(fred_data['rrp'].iloc[-1, 1])
    net_liquidity = walcl - tga - rrp
except: pass
try:
    margin_debt = float(fred_data['margin_debt'].iloc[-1, 1])
except: pass
try:
    fed_funds = float(fred_data['fedfunds'].iloc[-1, 1])
except: pass
try:
    m2_df = fred_data['m2']
    if m2_df is not None and len(m2_df) > 12:
        m2_df.columns = ['date', 'value']
        m2_df['value'] = pd.to_numeric(m2_df['value'], errors='coerce')
        m2_latest = m2_df['value'].iloc[-1]
        m2_year_ago = m2_df['value'].iloc[-13] if len(m2_df) >= 13 else m2_df['value'].iloc[0]
        m2_yoy = ((m2_latest - m2_year_ago) / m2_year_ago) * 100
except: pass
try:
    nfci = float(fred_data['nfci'].iloc[-1, 1])
except: pass
try:
    unemployment = float(fred_data['unrate'].iloc[-1, 1])
except: pass
try:
    consumer_sentiment = float(fred_data['umcsent'].iloc[-1, 1])
except: pass

# [B] YFinance 數據
@st.cache_data(ttl=1800)
def get_20d_momentum(tickers):
    mom_list = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period='25d')['Close'].dropna()
            if len(hist) >= 2:
                mom_list.append((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0])
        except: pass
    return sum(mom_list) / len(mom_list) if mom_list else 0.0

@st.cache_data(ttl=1800)
def get_latest_price(ticker):
    try:
        return yf.Ticker(ticker).history(period='5d')['Close'].dropna().iloc[-1]
    except:
        return None

@st.cache_data(ttl=1800)
def get_ticker_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

@st.cache_data(ttl=1800)
def get_valuation_metrics(tickers):
    metrics = []
    for t in tickers:
        info = get_ticker_info(t)
        metrics.append({
            'ticker': t,
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'pe_trailing': info.get('trailingPE'),
            'pe_forward': info.get('forwardPE'),
            'ps': info.get('priceToSalesTrailing12Months'),
            'pfcf': info.get('priceToFreeCashflows'),
            'peg': info.get('pegRatio'),
            'market_cap': info.get('marketCap'),
            'revenue_growth': info.get('revenueGrowth'),
            'profit_margin': info.get('profitMargins'),
        })
    return pd.DataFrame(metrics)

@st.cache_data(ttl=86400)
def get_hyperscaler_capex():
    capex_data = {}
    for ticker in HYPERSCALERS:
        try:
            t = yf.Ticker(ticker)
            cf = t.quarterly_cashflow
            inc = t.quarterly_income_stmt
            if cf is not None and inc is not None:
                capex_row = None
                for label in ['CapitalExpenditure', 'Capital Expenditure']:
                    if label in cf.index:
                        capex_row = cf.loc[label]
                        break
                rev_row = None
                for label in ['TotalRevenue', 'Total Revenue']:
                    if label in inc.index:
                        rev_row = inc.loc[label]
                        break
                if capex_row is not None and rev_row is not None:
                    dates = capex_row.index
                    capex_vals = capex_row.values
                    rev_vals = rev_row.reindex(dates).values
                    capex_data[ticker] = {
                        'dates': [d.strftime('%Y-%m-%d') for d in dates],
                        'capex': [abs(float(v)) / 1e9 if pd.notna(v) else None for v in capex_vals],
                        'revenue': [float(v) / 1e9 if pd.notna(v) else None for v in rev_vals],
                    }
        except: pass
    return capex_data

@st.cache_data(ttl=86400)
def get_historical_pe_spy():
    try:
        spy = yf.Ticker('SPY')
        hist = spy.history(period='10y', interval='1mo')['Close']
        return hist
    except:
        return None

# 抓取即時數據
vxn_val = get_latest_price('^VXN')
if vxn_val: vxn = vxn_val

tnx_val = get_latest_price('^TNX')
if tnx_val: tnx_yield = tnx_val

dxy_val = get_latest_price('DX-Y.NYB')
if dxy_val: dxy = dxy_val

# Put/Call ratio
try:
    pcr_val = get_latest_price('^PCCE')
    if pcr_val: put_call_ratio = pcr_val
except: pass

# 估值數據
val_df = get_valuation_metrics(TECH_GIANTS)
ev_ebitda_vals = val_df['ev_ebitda'].dropna()
avg_ev_ebitda = ev_ebitda_vals.mean() if len(ev_ebitda_vals) > 0 else 30.0

# 動能
avg_momentum = get_20d_momentum(TW_SUPPLY)
infra_momentum = get_20d_momentum(INFRA_STOCKS)
comm_momentum = get_20d_momentum(COMMODITIES)
sox_momentum = get_20d_momentum(['^SOX'])
btc_momentum = get_20d_momentum(['BTC-USD'])

# CapEx
capex_data = get_hyperscaler_capex()

# Buffett Indicator
buffett_indicator = None
try:
    wilshire_df = fred_data.get('wilshire')
    gdp_df = fred_data.get('gdp')
    if wilshire_df is not None and gdp_df is not None:
        latest_wilshire = float(wilshire_df.iloc[-1, 1])
        latest_gdp = float(gdp_df.iloc[-1, 1])
        buffett_indicator = (latest_wilshire / latest_gdp) * 100
except: pass

# =============================================================
# 2. 加權評估系統
# =============================================================
def normalize(value, safe_val, danger_val):
    score = ((value - safe_val) / (danger_val - safe_val)) * 100
    return min(100, max(0, score))

# X 軸：AI 動能 (加入 SOX 半導體 + BTC 風險偏好)
combined_growth = (
    avg_momentum * 0.35 +      # 台系代工
    infra_momentum * 0.25 +    # AI 基礎設施
    sox_momentum * 0.25 +      # 半導體指數
    btc_momentum * 0.15        # 風險偏好代理
)
growth_score = min(100, max(-100, combined_growth * 1000))

# Y 軸：12 因子壓力評估系統 (4 群組)
# ── Cluster A: 貨幣政策與流動性 (30%)
# ── Cluster B: 信用與金融條件 (20%)
# ── Cluster C: 估值與情緒 (25%)
# ── Cluster D: 總體經濟與通膨 (25%)

buffett_val = buffett_indicator if buffett_indicator else 130

risk_scores = {
    # Cluster A: 貨幣政策與流動性
    "10Y Treasury":     normalize(tnx_yield, 3.5, 5.0),
    "Fed Funds Rate":   normalize(fed_funds, 3.0, 5.5),
    "Net Liquidity":    normalize(6200 - net_liquidity, 0, 800),
    "M2 Supply YoY":    normalize(-m2_yoy, -6, 2),  # M2 shrinking = stress
    # Cluster B: 信用與金融條件
    "Yield Curve":      normalize(latest_curve, -0.8, 0.2),
    "Credit Spread":    normalize(latest_spread, 3.0, 5.5),
    "NFCI":             normalize(nfci, -0.5, 0.5),  # positive = tightening
    # Cluster C: 估值與情緒
    "EV/EBITDA":        normalize(avg_ev_ebitda, 25, 45),
    "Buffett Indicator": normalize(buffett_val, 100, 200),
    "VXN Volatility":   normalize(vxn, 15, 35),
    "Consumer Sentiment": normalize(100 - consumer_sentiment, 30, 70),  # low sentiment = stress
    # Cluster D: 總體經濟與通膨
    "US Dollar":        normalize(dxy, 100.0, 107.0),
    "Commodities":      normalize(comm_momentum, 0.0, 0.10),
    "Unemployment":     normalize(unemployment, 3.5, 6.0),
}

risk_labels_zh = {
    "10Y Treasury":      "定價引力",
    "Fed Funds Rate":    "聯準會利率",
    "Net Liquidity":     "絕對資金",
    "M2 Supply YoY":     "貨幣供給",
    "Yield Curve":       "衰退警報",
    "Credit Spread":     "違約風險",
    "NFCI":              "金融環境",
    "EV/EBITDA":         "估值極限",
    "Buffett Indicator": "巴菲特指標",
    "VXN Volatility":    "市場情緒",
    "Consumer Sentiment":"消費者信心",
    "US Dollar":         "全球水閘",
    "Commodities":       "通膨預期",
    "Unemployment":      "就業惡化",
}

risk_cluster = {
    "10Y Treasury":      "A 貨幣流動性",
    "Fed Funds Rate":    "A 貨幣流動性",
    "Net Liquidity":     "A 貨幣流動性",
    "M2 Supply YoY":     "A 貨幣流動性",
    "Yield Curve":       "B 信用條件",
    "Credit Spread":     "B 信用條件",
    "NFCI":              "B 信用條件",
    "EV/EBITDA":         "C 估值情緒",
    "Buffett Indicator": "C 估值情緒",
    "VXN Volatility":    "C 估值情緒",
    "Consumer Sentiment":"C 估值情緒",
    "US Dollar":         "D 總體通膨",
    "Commodities":       "D 總體通膨",
    "Unemployment":      "D 總體通膨",
}

weights = {
    # Cluster A: 30%
    "10Y Treasury":      0.08,
    "Fed Funds Rate":    0.08,
    "Net Liquidity":     0.08,
    "M2 Supply YoY":     0.06,
    # Cluster B: 20%
    "Yield Curve":       0.07,
    "Credit Spread":     0.07,
    "NFCI":              0.06,
    # Cluster C: 25%
    "EV/EBITDA":         0.07,
    "Buffett Indicator": 0.06,
    "VXN Volatility":    0.06,
    "Consumer Sentiment":0.06,
    # Cluster D: 25%
    "US Dollar":         0.09,
    "Commodities":       0.08,
    "Unemployment":      0.08,
}

stress_score = sum(risk_scores[k] * weights[k] for k in risk_scores)

if growth_score >= 0 and stress_score < 50:
    current_regime = "擴張期 Expansion"
    regime_color = "#2EA043"; regime_css = "safe"; regime_icon = "●"
elif growth_score >= 0 and stress_score >= 50:
    current_regime = "過熱期 Overheating"
    regime_color = "#D29922"; regime_css = "caution"; regime_icon = "▲"
elif growth_score < 0 and stress_score >= 50:
    current_regime = "泡沫破裂 Bust"
    regime_color = "#F85149"; regime_css = "danger"; regime_icon = "◆"
else:
    current_regime = "沉澱期 Contraction"
    regime_color = "#58A6FF"; regime_css = "cool"; regime_icon = "■"

# =============================================================
# 3. 歷史記錄存檔
# =============================================================
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_history.json")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except: pass

history = load_history()
now_str = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')

if not history or history[-1].get('time', '') != now_str:
    record = {
        'time': now_str,
        'stress_score': round(stress_score, 2),
        'growth_score': round(growth_score, 2),
        'regime': current_regime,
        'tnx': round(tnx_yield, 3),
        'dxy': round(dxy, 2),
        'net_liq': round(net_liquidity, 2),
        'curve': round(latest_curve, 3),
        'spread': round(latest_spread, 3),
        'ev_ebitda': round(avg_ev_ebitda, 2),
        'vxn': round(vxn, 2),
        'fed_funds': round(fed_funds, 3),
        'm2_yoy': round(m2_yoy, 2),
        'nfci': round(nfci, 3),
        'unemployment': round(unemployment, 2),
        'consumer_sentiment': round(consumer_sentiment, 1),
        'sox_mom': round(sox_momentum * 100, 2),
        'btc_mom': round(btc_momentum * 100, 2),
    }
    if buffett_indicator:
        record['buffett'] = round(buffett_indicator, 2)
    if put_call_ratio:
        record['pcr'] = round(put_call_ratio, 3)
    if margin_debt:
        record['margin_debt'] = round(margin_debt, 2)

    history.append(record)
    if len(history) > 2000:
        history = history[-2000:]
    save_history(history)

# =============================================================
# 4. 頁面渲染
# =============================================================
# --- Title ---
st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
    <h1 style="margin:0; border:none !important; padding:0;">Global Macro Regime Radar</h1>
    <span style="font-size:0.7rem; color:#484F58; background:#161B22; padding:3px 10px; border-radius:9999px; border:1px solid #21262D;
                 font-family:'JetBrains Mono',monospace; font-weight:500;">v6.0</span>
</div>
<p style="color:#484F58; font-size:0.75rem; font-family:'JetBrains Mono',monospace; margin:0 0 20px 0;">
    {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC &nbsp;·&nbsp; FRED &nbsp;·&nbsp; Yahoo Finance
</p>
""", unsafe_allow_html=True)

# --- Regime Banner ---
st.markdown(f"""
<div class="regime-banner {regime_css}">
    <div class="regime-flex">
        <div class="regime-left">
            <div class="regime-dot" style="color:{regime_color}; background:{regime_color};"></div>
            <div>
                <div class="regime-title">{current_regime}</div>
                <div class="regime-sub">AI Infrastructure & Macro Liquidity Assessment</div>
            </div>
        </div>
        <div class="regime-right">
            <div class="regime-score" style="color:{regime_color};">{stress_score:.1f}</div>
            <div class="regime-label">Weighted Stress Score</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- KPI Cards Row 1 ---
kpi_r1 = st.columns(7)
kpi_data_r1 = [
    ("10Y TREASURY", f"{tnx_yield:.2f}%"),
    ("FED FUNDS", f"{fed_funds:.2f}%"),
    ("US DOLLAR", f"{dxy:.2f}"),
    ("NET LIQUIDITY", f"${net_liquidity/1000:.2f}T"),
    ("M2 YoY", f"{m2_yoy:+.1f}%"),
    ("YIELD CURVE", f"{latest_curve:.2f}%"),
    ("NFCI", f"{nfci:+.2f}"),
]
for i, (label, value) in enumerate(kpi_data_r1):
    kpi_r1[i].metric(label, value)

# --- KPI Cards Row 2 ---
kpi_r2 = st.columns(7)
kpi_data_r2 = [
    ("CREDIT SPREAD", f"{latest_spread:.2f}%"),
    ("VXN INDEX", f"{vxn:.1f}"),
    ("UNEMPLOYMENT", f"{unemployment:.1f}%"),
    ("CONSUMER SENT.", f"{consumer_sentiment:.0f}"),
    ("SOX MOMENTUM", f"{sox_momentum*100:+.1f}%"),
    ("BTC MOMENTUM", f"{btc_momentum*100:+.1f}%"),
    ("BUFFETT IND.", f"{buffett_val:.0f}%"),
]
for i, (label, value) in enumerate(kpi_data_r2):
    kpi_r2[i].metric(label, value)

st.markdown("---")

# =============================================================
# 第一排：Regime Map + Risk Radar
# =============================================================
st.markdown("## REGIME MAP & RISK DECOMPOSITION")
col_map, col_radar = st.columns([1.3, 1])

with col_map:
    fig_scatter = go.Figure()
    fig_scatter.add_shape(type="rect", x0=0, y0=50, x1=100, y1=100,
                          fillcolor="rgba(210,153,34,0.06)", line_width=0)
    fig_scatter.add_shape(type="rect", x0=0, y0=0, x1=100, y1=50,
                          fillcolor="rgba(46,160,67,0.06)", line_width=0)
    fig_scatter.add_shape(type="rect", x0=-100, y0=50, x1=0, y1=100,
                          fillcolor="rgba(248,81,73,0.06)", line_width=0)
    fig_scatter.add_shape(type="rect", x0=-100, y0=0, x1=0, y1=50,
                          fillcolor="rgba(88,166,255,0.06)", line_width=0)

    fig_scatter.add_hline(y=50, line_dash="dot", line_color="#30363D", line_width=1)
    fig_scatter.add_vline(x=0, line_dash="dot", line_color="#30363D", line_width=1)

    annotations_map = [
        dict(x=50, y=95, text="OVERHEATING", showarrow=False, font=dict(size=10, color="#484F58")),
        dict(x=50, y=25, text="EXPANSION", showarrow=False, font=dict(size=10, color="#484F58")),
        dict(x=-50, y=95, text="BUST", showarrow=False, font=dict(size=10, color="#484F58")),
        dict(x=-50, y=25, text="CONTRACTION", showarrow=False, font=dict(size=10, color="#484F58")),
    ]

    fig_scatter.add_trace(go.Scatter(
        x=[growth_score], y=[stress_score], mode='markers+text',
        marker=dict(size=18, color=regime_color,
                    line=dict(width=2, color='white'),
                    symbol='circle'),
        text=[f"Stress: {stress_score:.1f}"],
        textposition="top center",
        textfont=dict(color="#E6EDF3", size=12, family="JetBrains Mono"),
        hovertemplate=f"Growth: {growth_score:.1f}<br>Stress: {stress_score:.1f}<extra></extra>"
    ))

    if len(history) > 1:
        trail_x = [h['growth_score'] for h in history[-10:]]
        trail_y = [h['stress_score'] for h in history[-10:]]
        fig_scatter.add_trace(go.Scatter(
            x=trail_x, y=trail_y, mode='lines',
            line=dict(color=regime_color, width=1, dash='dot'),
            opacity=0.3, showlegend=False,
            hoverinfo='skip'
        ))

    apply_theme(fig_scatter, title_text="Macro Regime Map",
        xaxis_title="AI Growth Momentum",
        yaxis_title="Weighted Stress",
        xaxis=dict(**PLOTLY_AXIS['xaxis'], range=[-100, 100]),
        yaxis=dict(**PLOTLY_AXIS['yaxis'], range=[-10, 100]),
        height=420,
        annotations=annotations_map,
        showlegend=False
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_radar:
    categories = [f"{k}\n({risk_labels_zh[k]})" for k in risk_scores.keys()]
    radar_vals = list(risk_scores.values())

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[30]*len(categories) + [30],
        theta=categories + [categories[0]],
        fill='toself', fillcolor="rgba(46,160,67,0.05)",
        line=dict(color="#21262D", dash="dash", width=1),
        name="Safe Zone", hoverinfo='skip'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor=f"rgba({int(regime_color[1:3],16)},{int(regime_color[3:5],16)},{int(regime_color[5:7],16)},0.2)",
        line=dict(color=regime_color, width=2),
        name="Current Risk"
    ))
    apply_theme(fig_radar, title_text="14-Factor Risk Decomposition",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#21262D",
                           tickfont=dict(size=9, color="#484F58")),
            angularaxis=dict(gridcolor="#21262D",
                            tickfont=dict(size=10, color="#8B949E")),
            bgcolor="#161B22"
        ),
        showlegend=False, height=420,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# =============================================================
# 群組壓力分解 (Cluster Breakdown)
# =============================================================
st.markdown("## CLUSTER STRESS DECOMPOSITION")
st.markdown("<p style='color:#8B949E; font-size:0.8rem; margin-top:-10px;'>14 因子分 4 群組 — 各群組加權壓力貢獻</p>", unsafe_allow_html=True)

cluster_scores = {}
for k in risk_scores:
    cl = risk_cluster[k]
    if cl not in cluster_scores:
        cluster_scores[cl] = {'total': 0, 'weight_sum': 0, 'factors': []}
    cluster_scores[cl]['total'] += risk_scores[k] * weights[k]
    cluster_scores[cl]['weight_sum'] += weights[k]
    cluster_scores[cl]['factors'].append((k, risk_scores[k], weights[k]))

cluster_cols = st.columns(4)
cluster_colors = {"A 貨幣流動性": "#58A6FF", "B 信用條件": "#BC8CFF", "C 估值情緒": "#F0883E", "D 總體通膨": "#39D2C0"}
for i, (cl_name, cl_data) in enumerate(sorted(cluster_scores.items())):
    weighted_avg = cl_data['total'] / cl_data['weight_sum'] if cl_data['weight_sum'] > 0 else 0
    cl_color = cluster_colors.get(cl_name, "#8B949E")
    badge = "safe" if weighted_avg < 35 else ("caution" if weighted_avg < 65 else "danger")
    with cluster_cols[i]:
        st.markdown(f"""
        <div style="background:#161B22; border:1px solid #21262D; border-radius:10px; padding:16px 20px; height:100%;">
            <div style="font-size:0.7rem; color:#8B949E; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:8px;">
                {cl_name}
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.8rem; font-weight:700; color:{cl_color};">
                {weighted_avg:.0f}
            </div>
            <div style="font-size:0.7rem; color:#484F58; margin-top:4px;">權重 {cl_data['weight_sum']*100:.0f}% · 貢獻 {cl_data['total']:.1f}pt</div>
            <div style="margin-top:10px; border-top:1px solid #21262D; padding-top:8px;">
        """, unsafe_allow_html=True)
        for fname, fscore, fweight in cl_data['factors']:
            sc_color = "#2EA043" if fscore < 30 else ("#D29922" if fscore < 65 else "#F85149")
            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; font-size:0.72rem; margin:3px 0; color:#8B949E;">
                    <span>{risk_labels_zh[fname]}</span>
                    <span style="font-family:'JetBrains Mono',monospace; color:{sc_color};">{fscore:.0f}</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

st.markdown("---")

# =============================================================
# 貨幣政策與金融條件趨勢
# =============================================================
st.markdown("## MONETARY POLICY & FINANCIAL CONDITIONS")
col_mon1, col_mon2, col_mon3 = st.columns(3)

with col_mon1:
    ff_df = fred_data.get('fedfunds')
    if ff_df is not None:
        ff_plot = ff_df.copy()
        ff_plot.columns = ['date', 'value']
        ff_plot['date'] = pd.to_datetime(ff_plot['date'])
        ff_plot['value'] = pd.to_numeric(ff_plot['value'], errors='coerce')
        ff_plot = ff_plot.dropna().tail(120)
        fig_ff = go.Figure()
        fig_ff.add_trace(go.Scatter(
            x=ff_plot['date'], y=ff_plot['value'],
            mode='lines', fill='tozeroy',
            fillcolor="rgba(88,166,255,0.08)",
            line=dict(color="#58A6FF", width=2), name="Fed Funds Rate"
        ))
        apply_theme(fig_ff, title_text="Fed Funds Rate (%)", height=300, showlegend=False)
        st.plotly_chart(fig_ff, use_container_width=True)
    else:
        st.info("Fed Funds Rate 數據載入中...")

with col_mon2:
    m2_raw = fred_data.get('m2')
    if m2_raw is not None:
        m2_plot = m2_raw.copy()
        m2_plot.columns = ['date', 'value']
        m2_plot['date'] = pd.to_datetime(m2_plot['date'])
        m2_plot['value'] = pd.to_numeric(m2_plot['value'], errors='coerce')
        m2_plot = m2_plot.dropna().tail(120)
        m2_plot['yoy'] = m2_plot['value'].pct_change(12) * 100
        m2_yoy_plot = m2_plot.dropna(subset=['yoy'])
        fig_m2 = go.Figure()
        fig_m2.add_trace(go.Scatter(
            x=m2_yoy_plot['date'], y=m2_yoy_plot['yoy'],
            mode='lines', fill='tozeroy',
            fillcolor="rgba(188,140,255,0.08)",
            line=dict(color="#BC8CFF", width=2), name="M2 YoY %"
        ))
        fig_m2.add_hline(y=0, line_dash="dash", line_color="#F85149", line_width=1)
        apply_theme(fig_m2, title_text="M2 Money Supply YoY (%)", height=300, showlegend=False)
        st.plotly_chart(fig_m2, use_container_width=True)
    else:
        st.info("M2 貨幣供給數據載入中...")

with col_mon3:
    nfci_df = fred_data.get('nfci')
    if nfci_df is not None:
        nfci_plot = nfci_df.copy()
        nfci_plot.columns = ['date', 'value']
        nfci_plot['date'] = pd.to_datetime(nfci_plot['date'])
        nfci_plot['value'] = pd.to_numeric(nfci_plot['value'], errors='coerce')
        nfci_plot = nfci_plot.dropna().tail(120)
        fig_nfci = go.Figure()
        fig_nfci.add_hrect(y0=0, y1=nfci_plot['value'].max()+0.2, fillcolor="rgba(248,81,73,0.04)", line_width=0)
        fig_nfci.add_hrect(y0=nfci_plot['value'].min()-0.2, y1=0, fillcolor="rgba(46,160,67,0.04)", line_width=0)
        fig_nfci.add_trace(go.Scatter(
            x=nfci_plot['date'], y=nfci_plot['value'],
            mode='lines',
            line=dict(color="#39D2C0", width=2), name="NFCI"
        ))
        fig_nfci.add_hline(y=0, line_dash="dash", line_color="#484F58", line_width=1,
                          annotation_text="Neutral", annotation_font=dict(size=9, color="#484F58"))
        apply_theme(fig_nfci, title_text="Chicago Fed NFCI", height=300, showlegend=False)
        st.plotly_chart(fig_nfci, use_container_width=True)
    else:
        st.info("NFCI 數據載入中...")

st.markdown("---")

# =============================================================
# 消費者信心 + 失業率
# =============================================================
st.markdown("## MACRO HEALTH INDICATORS")
col_sent, col_unemp = st.columns(2)

with col_sent:
    sent_df = fred_data.get('umcsent')
    if sent_df is not None:
        sent_plot = sent_df.copy()
        sent_plot.columns = ['date', 'value']
        sent_plot['date'] = pd.to_datetime(sent_plot['date'])
        sent_plot['value'] = pd.to_numeric(sent_plot['value'], errors='coerce')
        sent_plot = sent_plot.dropna().tail(120)
        fig_sent = go.Figure()
        fig_sent.add_hrect(y0=0, y1=60, fillcolor="rgba(248,81,73,0.04)", line_width=0)
        fig_sent.add_trace(go.Scatter(
            x=sent_plot['date'], y=sent_plot['value'],
            mode='lines', fill='tozeroy',
            fillcolor="rgba(240,136,62,0.08)",
            line=dict(color="#F0883E", width=2), name="UMich Sentiment"
        ))
        fig_sent.add_hline(y=60, line_dash="dash", line_color="#F85149",
                          annotation_text="Recession Zone (<60)",
                          annotation_font=dict(size=9, color="#F85149"))
        apply_theme(fig_sent, title_text="U. Michigan Consumer Sentiment", height=340, showlegend=False)
        st.plotly_chart(fig_sent, use_container_width=True)
    else:
        st.info("消費者信心數據載入中...")

with col_unemp:
    unemp_df = fred_data.get('unrate')
    if unemp_df is not None:
        unemp_plot = unemp_df.copy()
        unemp_plot.columns = ['date', 'value']
        unemp_plot['date'] = pd.to_datetime(unemp_plot['date'])
        unemp_plot['value'] = pd.to_numeric(unemp_plot['value'], errors='coerce')
        unemp_plot = unemp_plot.dropna().tail(120)
        fig_unemp = go.Figure()
        fig_unemp.add_hrect(y0=5.0, y1=unemp_plot['value'].max()+1, fillcolor="rgba(248,81,73,0.04)", line_width=0)
        fig_unemp.add_trace(go.Scatter(
            x=unemp_plot['date'], y=unemp_plot['value'],
            mode='lines', fill='tozeroy',
            fillcolor="rgba(247,120,186,0.08)",
            line=dict(color="#F778BA", width=2), name="Unemployment"
        ))
        fig_unemp.add_hline(y=5.0, line_dash="dash", line_color="#D29922",
                          annotation_text="Stress Threshold (5%+)",
                          annotation_font=dict(size=9, color="#D29922"))
        apply_theme(fig_unemp, title_text="U.S. Unemployment Rate (%)", height=340, showlegend=False)
        st.plotly_chart(fig_unemp, use_container_width=True)
    else:
        st.info("失業率數據載入中...")

st.markdown("---")

# =============================================================
# 估值歷史百分位 + CAPE / Buffett Indicator
# =============================================================
st.markdown("## VALUATION ANALYSIS")
col_val_table, col_gauges = st.columns([1.5, 1])

with col_val_table:
    st.markdown("#### AI 巨頭估值矩陣")
    if not val_df.empty:
        display_df = val_df.copy()
        display_df.columns = ['Ticker', 'EV/EBITDA', 'P/E (TTM)', 'P/E (Fwd)',
                              'P/S', 'P/FCF', 'PEG', 'Market Cap', 'Rev Growth', 'Profit Margin']

        for col in ['EV/EBITDA', 'P/E (TTM)', 'P/E (Fwd)', 'P/S', 'P/FCF', 'PEG']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
        display_df['Market Cap'] = display_df['Market Cap'].apply(
            lambda x: f"${x/1e12:.2f}T" if pd.notna(x) and x > 1e12 else (f"${x/1e9:.0f}B" if pd.notna(x) else "N/A"))
        display_df['Rev Growth'] = display_df['Rev Growth'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        display_df['Profit Margin'] = display_df['Profit Margin'].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    pe_vals = val_df['pe_trailing'].dropna()
    ps_vals = val_df['ps'].dropna()
    avg_pe = pe_vals.mean() if len(pe_vals) > 0 else None
    avg_ps = ps_vals.mean() if len(ps_vals) > 0 else None

    pe_percentile = min(100, max(0, (avg_pe - 15) / (60 - 15) * 100)) if avg_pe else None
    ps_percentile = min(100, max(0, (avg_ps - 5) / (30 - 5) * 100)) if avg_ps else None
    ev_percentile = min(100, max(0, (avg_ev_ebitda - 15) / (50 - 15) * 100))

    st.markdown("#### 估值歷史百分位估算")
    pcol1, pcol2, pcol3 = st.columns(3)
    if pe_percentile is not None:
        pcol1.metric("P/E 百分位", f"{pe_percentile:.0f}th", delta=f"Avg: {avg_pe:.1f}x")
    if ps_percentile is not None:
        pcol2.metric("P/S 百分位", f"{ps_percentile:.0f}th", delta=f"Avg: {avg_ps:.1f}x")
    pcol3.metric("EV/EBITDA 百分位", f"{ev_percentile:.0f}th", delta=f"Avg: {avg_ev_ebitda:.1f}x")

with col_gauges:
    st.markdown("#### 泡沫經典指標")

    # Buffett Indicator Gauge
    fig_gauges = make_subplots(rows=2, cols=1,
                               specs=[[{"type": "indicator"}], [{"type": "indicator"}]],
                               vertical_spacing=0.3)

    buffett_val = buffett_indicator if buffett_indicator else 0
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=buffett_val,
        title={'text': "Buffett Indicator<br><span style='font-size:0.7em;color:#8B949E'>Wilshire 5000 / GDP × 100</span>",
               'font': {'size': 14, 'color': '#E6EDF3'}},
        number={'font': {'family': 'JetBrains Mono', 'size': 28, 'color': '#E6EDF3'}, 'suffix': '%'},
        gauge=dict(
            axis=dict(range=[50, 250], tickfont=dict(size=10, color="#484F58")),
            bar=dict(color=regime_color, thickness=0.3),
            bgcolor="#21262D",
            borderwidth=0,
            steps=[
                dict(range=[50, 100], color="rgba(46,160,67,0.15)"),
                dict(range=[100, 150], color="rgba(57,210,192,0.15)"),
                dict(range=[150, 200], color="rgba(210,153,34,0.15)"),
                dict(range=[200, 250], color="rgba(248,81,73,0.15)"),
            ],
            threshold=dict(line=dict(color="#F85149", width=2), thickness=0.8, value=200)
        )
    ), row=1, col=1)

    # S&P 500 EV/EBITDA Gauge (as proxy for CAPE)
    fig_gauges.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_ev_ebitda,
        title={'text': "AI Sector EV/EBITDA<br><span style='font-size:0.7em;color:#8B949E'>Top 5 平均估值倍數</span>",
               'font': {'size': 14, 'color': '#E6EDF3'}},
        number={'font': {'family': 'JetBrains Mono', 'size': 28, 'color': '#E6EDF3'}, 'suffix': 'x'},
        gauge=dict(
            axis=dict(range=[10, 60], tickfont=dict(size=10, color="#484F58")),
            bar=dict(color=regime_color, thickness=0.3),
            bgcolor="#21262D",
            borderwidth=0,
            steps=[
                dict(range=[10, 20], color="rgba(46,160,67,0.15)"),
                dict(range=[20, 30], color="rgba(57,210,192,0.15)"),
                dict(range=[30, 40], color="rgba(210,153,34,0.15)"),
                dict(range=[40, 60], color="rgba(248,81,73,0.15)"),
            ],
            threshold=dict(line=dict(color="#F85149", width=2), thickness=0.8, value=45)
        )
    ), row=2, col=1)

    fig_gauges.update_layout(
        paper_bgcolor="#161B22", font=dict(color="#8B949E"),
        height=450, margin=dict(l=30, r=30, t=30, b=10)
    )
    st.plotly_chart(fig_gauges, use_container_width=True)

st.markdown("---")

# =============================================================
# 第三排：AI CapEx 追蹤
# =============================================================
st.markdown("## AI INFRASTRUCTURE CAPEX TRACKER")
st.markdown("<p style='color:#8B949E; font-size:0.8rem; margin-top:-10px;'>超大型雲端業者季度資本支出 — AI 軍備競賽的彈藥量</p>", unsafe_allow_html=True)

if capex_data:
    col_capex, col_pct = st.columns(2)

    with col_capex:
        fig_capex = go.Figure()
        for i, (ticker, data) in enumerate(capex_data.items()):
            dates = pd.to_datetime(data['dates'])
            capex_vals = data['capex']
            sorted_pairs = sorted(zip(dates, capex_vals))
            dates_sorted = [p[0] for p in sorted_pairs]
            capex_sorted = [p[1] for p in sorted_pairs]

            fig_capex.add_trace(go.Bar(
                x=dates_sorted, y=capex_sorted,
                name=ticker,
                marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                opacity=0.85
            ))

        apply_theme(fig_capex, title_text="Quarterly CapEx ($B)",
            barmode='group',
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                       font=dict(size=11, color="#8B949E"), bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_capex, use_container_width=True)

    with col_pct:
        fig_pct = go.Figure()
        for i, (ticker, data) in enumerate(capex_data.items()):
            dates = pd.to_datetime(data['dates'])
            capex_vals = data['capex']
            rev_vals = data['revenue']
            pct_vals = []
            valid_dates = []
            for d, c, r in sorted(zip(dates, capex_vals, rev_vals)):
                if c is not None and r is not None and r > 0:
                    pct_vals.append(c / r * 100)
                    valid_dates.append(d)

            if pct_vals:
                fig_pct.add_trace(go.Scatter(
                    x=valid_dates, y=pct_vals,
                    name=ticker, mode='lines+markers',
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                    marker=dict(size=6)
                ))

        fig_pct.add_hline(y=25, line_dash="dash", line_color="#F85149",
                         annotation_text="Danger Zone (25%+)",
                         annotation_font=dict(size=10, color="#F85149"))

        apply_theme(fig_pct, title_text="CapEx as % of Revenue",
            yaxis_title="%",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                       font=dict(size=11, color="#8B949E"), bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_pct, use_container_width=True)

    total_capex = sum(
        sum(v for v in data['capex'] if v is not None)
        for data in capex_data.values()
    )
    latest_q_capex = sum(
        data['capex'][list(pd.to_datetime(data['dates'])).index(max(pd.to_datetime(data['dates'])))]
        for data in capex_data.values()
        if data['capex']
    )
    st.markdown(f"""
    <div style="background:#161B22; border:1px solid #21262D; border-radius:10px; padding:16px 24px; margin:8px 0;">
        <span style="color:#8B949E; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em;">
            Latest Quarter Combined CapEx
        </span>
        <span style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:600; color:#E6EDF3; margin-left:16px;">
            ${latest_q_capex:.1f}B
        </span>
        <span style="color:#484F58; font-size:0.75rem; margin-left:12px;">
            (MSFT + GOOG + AMZN + META)
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("CapEx 數據載入中或暫時無法取得...")

st.markdown("---")

# =============================================================
# 第四排：壓力分數歷史走勢
# =============================================================
st.markdown("## STRESS SCORE HISTORY")

if len(history) > 1:
    hist_df = pd.DataFrame(history)
    hist_df['time'] = pd.to_datetime(hist_df['time'])

    fig_history = go.Figure()

    fig_history.add_hrect(y0=0, y1=25, fillcolor="rgba(46,160,67,0.04)", line_width=0)
    fig_history.add_hrect(y0=25, y1=50, fillcolor="rgba(57,210,192,0.04)", line_width=0)
    fig_history.add_hrect(y0=50, y1=75, fillcolor="rgba(210,153,34,0.04)", line_width=0)
    fig_history.add_hrect(y0=75, y1=100, fillcolor="rgba(248,81,73,0.04)", line_width=0)

    fig_history.add_trace(go.Scatter(
        x=hist_df['time'], y=hist_df['stress_score'],
        mode='lines',
        fill='tozeroy',
        fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="#58A6FF", width=2),
        name="Stress Score",
        hovertemplate="%{x}<br>Stress: %{y:.1f}<extra></extra>"
    ))

    if len(hist_df) > 0:
        last_point = hist_df.iloc[-1]
        fig_history.add_trace(go.Scatter(
            x=[last_point['time']], y=[last_point['stress_score']],
            mode='markers',
            marker=dict(size=10, color=regime_color, line=dict(width=2, color='white')),
            showlegend=False,
            hoverinfo='skip'
        ))

    fig_history.add_hline(y=50, line_dash="dash", line_color="#D29922",
                         annotation_text="Caution Threshold",
                         annotation_font=dict(size=10, color="#D29922"),
                         annotation_position="bottom right")
    fig_history.add_hline(y=75, line_dash="dash", line_color="#F85149",
                         annotation_text="Danger Threshold",
                         annotation_font=dict(size=10, color="#F85149"),
                         annotation_position="bottom right")

    apply_theme(fig_history, title_text="Weighted Stress Score Over Time",
        yaxis=dict(**PLOTLY_AXIS['yaxis'], range=[0, 100]),
        height=350, showlegend=False
    )
    st.plotly_chart(fig_history, use_container_width=True)
else:
    st.info("歷史數據將在多次執行後開始顯示趨勢圖。首次執行已記錄第一筆資料。")

st.markdown("---")

# =============================================================
# 第五排：Margin Debt + Put/Call Ratio
# =============================================================
st.markdown("## MARKET SENTIMENT & LEVERAGE")
col_margin, col_pcr = st.columns(2)

with col_margin:
    if fred_data.get('margin_debt') is not None:
        md_df = fred_data['margin_debt'].copy()
        md_df.columns = ['date', 'value']
        md_df['date'] = pd.to_datetime(md_df['date'])
        md_df['value'] = pd.to_numeric(md_df['value'], errors='coerce')
        md_df = md_df.dropna().tail(40)

        fig_md = go.Figure()
        fig_md.add_trace(go.Bar(
            x=md_df['date'], y=md_df['value'] / 1000,
            marker_color="#58A6FF", opacity=0.7, name="Margin Debt"
        ))

        if len(md_df) >= 4:
            md_df['ma'] = md_df['value'].rolling(4).mean()
            fig_md.add_trace(go.Scatter(
                x=md_df['date'], y=md_df['ma'] / 1000,
                mode='lines', line=dict(color="#BC8CFF", width=2),
                name="4Q Moving Avg"
            ))

        apply_theme(fig_md, title_text="FINRA Margin Debt ($B)",
            yaxis_title="$B",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                       font=dict(size=11, color="#8B949E"), bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_md, use_container_width=True)
    else:
        st.info("保證金負債數據暫時無法取得 (FRED)")

with col_pcr:
    if put_call_ratio is not None:
        try:
            pcr_hist = yf.Ticker('^PCCE').history(period='6mo')['Close'].dropna()

            fig_pcr = go.Figure()
            fig_pcr.add_trace(go.Scatter(
                x=pcr_hist.index, y=pcr_hist.values,
                mode='lines', fill='tozeroy',
                fillcolor="rgba(57,210,192,0.08)",
                line=dict(color="#39D2C0", width=2),
                name="Put/Call Ratio"
            ))

            fig_pcr.add_hline(y=1.0, line_dash="dash", line_color="#F85149",
                             annotation_text="Fear Zone (>1.0)",
                             annotation_font=dict(size=10, color="#F85149"))
            fig_pcr.add_hline(y=0.7, line_dash="dash", line_color="#2EA043",
                             annotation_text="Complacency (<0.7)",
                             annotation_font=dict(size=10, color="#2EA043"))

            apply_theme(fig_pcr, title_text="CBOE Equity Put/Call Ratio",
                height=380, showlegend=False
            )
            st.plotly_chart(fig_pcr, use_container_width=True)
        except:
            st.info("Put/Call Ratio 歷史圖表載入失敗，當前值: {:.3f}".format(put_call_ratio))
    else:
        fig_pcr_placeholder = go.Figure()
        fig_pcr_placeholder.add_annotation(
            text="Put/Call Ratio Data Unavailable<br>Will attempt ^PCCE from Yahoo Finance",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#484F58")
        )
        apply_theme(fig_pcr_placeholder,
            height=380,
            xaxis=dict(visible=False), yaxis=dict(visible=False)
        )
        st.plotly_chart(fig_pcr_placeholder, use_container_width=True)

st.markdown("---")

# =============================================================
# 綜合壓力評估矩陣
# =============================================================
st.markdown("## RISK ASSESSMENT MATRIX")

with st.expander("展開查看綜合壓力評估矩陣詳細計算", expanded=False):
    matrix_data = []
    for k in risk_scores:
        score = risk_scores[k]
        weight = weights[k]
        badge_class = "safe" if score < 30 else ("caution" if score < 65 else "danger")
        matrix_data.append({
            "風險維度": f"{k} ({risk_labels_zh[k]})",
            "風險分數": f"{score:.1f}",
            "配置權重": f"{weight*100:.0f}%",
            "加權貢獻": f"{score * weight:.1f}",
            "風險等級": "低" if score < 30 else ("中" if score < 65 else "高")
        })
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

# =============================================================
# 分析報告
# =============================================================
st.markdown("## ANALYSIS & RECOMMENDATION")

primary_risk_name = max(risk_scores, key=risk_scores.get)
primary_risk_value = risk_scores[primary_risk_name]

st.markdown(f"""
<div style="background:#161B22; border:1px solid #21262D; border-radius:10px; padding:20px 28px;">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
        <span class="risk-badge {regime_css}">{regime_icon} {current_regime}</span>
        <span style="color:#484F58; font-size:0.75rem;">Weighted Stress: {stress_score:.1f}/100 &nbsp;|&nbsp; Growth Momentum: {growth_score:.1f}</span>
    </div>
""", unsafe_allow_html=True)

# 找出壓力最高的群組
worst_cluster = max(cluster_scores.items(), key=lambda x: x[1]['total'] / x[1]['weight_sum'] if x[1]['weight_sum'] > 0 else 0)
worst_cluster_name = worst_cluster[0]
worst_cluster_avg = worst_cluster[1]['total'] / worst_cluster[1]['weight_sum']

# AI Growth 成分分解
growth_components = []
if sox_momentum != 0: growth_components.append(f"SOX半導體 {sox_momentum*100:+.1f}%")
if btc_momentum != 0: growth_components.append(f"BTC風險偏好 {btc_momentum*100:+.1f}%")
if avg_momentum != 0: growth_components.append(f"台系代工 {avg_momentum*100:+.1f}%")
growth_detail = "、".join(growth_components) if growth_components else "數據取得中"

if "擴張期" in current_regime:
    st.info(f"宏觀 14 因子均處於溫和區間，AI 基礎設施動能強勁（{growth_detail}）。Fed Funds {fed_funds:.2f}%、M2 YoY {m2_yoy:+.1f}%，資金環境健康。建議維持多頭配置。")
elif "過熱期" in current_regime:
    st.warning(f"⚠️ AI 動能仍在支撐大盤（{growth_detail}），但「{worst_cluster_name}」群組壓力已達 {worst_cluster_avg:.0f}/100。最大單一風險源：**{primary_risk_name} ({risk_labels_zh[primary_risk_name]})** = {primary_risk_value:.0f}。"
               f"\n\n消費者信心 {consumer_sentiment:.0f}、失業率 {unemployment:.1f}%、NFCI {nfci:+.2f}。建議開始對沖風險。")
elif "泡沫破裂" in current_regime:
    st.error(f"🛑 AI 供應鏈動能已疲態（{growth_detail}），同時「{worst_cluster_name}」群組壓力 {worst_cluster_avg:.0f}/100 達到極值。"
             f"\n\n最大風險源：**{primary_risk_name}**。Fed Funds {fed_funds:.2f}%、NFCI {nfci:+.2f}、失業率 {unemployment:.1f}%。請嚴格控制現金水位！")
elif "沉澱期" in current_regime:
    st.success(f"總體壓力見頂回落（「{worst_cluster_name}」{worst_cluster_avg:.0f}/100），AI 動能打底中（{growth_detail}）。M2 YoY {m2_yoy:+.1f}%，流動性正在回穩。長線佈局視窗開啟。")

if buffett_indicator:
    if buffett_indicator > 200:
        st.warning(f"📊 Buffett Indicator 達 {buffett_indicator:.0f}%，遠超歷史平均 (>200% 為極端高估)。")
    elif buffett_indicator > 150:
        st.info(f"📊 Buffett Indicator 為 {buffett_indicator:.0f}%，處於偏高區間 (歷史均值約 100%)。")

if consumer_sentiment < 55:
    st.warning(f"🧠 消費者信心指數 {consumer_sentiment:.0f}，低於 55 為衰退預警區。歷史上此水平常伴隨經濟放緩。")

if nfci > 0.2:
    st.warning(f"🏦 芝加哥金融條件指數 NFCI = {nfci:+.2f}，正值代表金融條件收緊中，信貸壓力增加。")

if m2_yoy < -2:
    st.error(f"💰 M2 貨幣供給 YoY = {m2_yoy:+.1f}%，罕見的負成長代表流動性正在收縮，歷史上高度關聯資產價格下跌。")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center; padding:24px 0 8px 0; color:#30363D; font-size:0.7rem; font-family:'JetBrains Mono',monospace;">
    AI Bubble Monitor v6.0 &nbsp;·&nbsp; Data delayed 15-20 min &nbsp;·&nbsp; Not financial advice &nbsp;·&nbsp; {datetime.datetime.utcnow().strftime('%Y-%m-%d')}
</div>
""", unsafe_allow_html=True)
