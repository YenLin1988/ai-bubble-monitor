import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime
import warnings
warnings.filterwarnings('ignore')

# 1. 設定監控對象
TECH_GIANTS = ['NVDA', 'MSFT', 'META', 'GOOG']
TW_SUPPLY_CHAIN = ['TSM', 'UMC', 'ASX'] # 台灣半導體三雄美股 ADR
TODAY = datetime.date.today()

def scale_score(value, min_val, max_val, invert=False):
    """將數值標準化至 0-100 分"""
    if value is None or pd.isna(value): return 50
    score = ((value - min_val) / (max_val - min_val)) * 100
    score = max(0, min(100, score))
    return 100 - score if invert else score

def get_status_color_and_message(score):
    """根據分數回傳顏色區間與對應訊息"""
    if score >= 75:
        return "🔴 紅色危險警報區 (75-100)", "極度危險 (Extreme Bubble) - 估值嚴重脫離基本面，流動性枯竭。建議全面落袋為安，啟動避險策略！"
    elif score >= 50:
        return "🟡 黃色警示區 (50-74)", "高度狂熱 (High Hype) - 市場情緒過熱，估值偏高。請密切緊盯供應鏈是否出現砍單或衰退訊號。"
    else:
        return "🟢 綠色安全區 (0-49)", "健康擴張 (Healthy) - 基本面有強烈支撐，資金流動性充裕，泡沫破裂風險低。"

def run_quant_model():
    print(f"==================================================")
    print(f" 執行時間 (UTC): {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"==================================================")
    
    scores = {}
    
    # ---- 模組一：總體流動性 (權重 30%) ----
    try:
        macro_data = web.DataReader(['BAMLH0A0HYM2', 'T10Y2Y'], 'fred', TODAY - datetime.timedelta(days=10), TODAY)
        latest_spread = macro_data['BAMLH0A0HYM2'].dropna().iloc[-1]
        latest_curve = macro_data['T10Y2Y'].dropna().iloc[-1]
        
        scores['Credit_Spread'] = scale_score(latest_spread, 3.0, 5.5)
        scores['Yield_Curve'] = scale_score(latest_curve, -0.8, 0.2)
    except:
        scores['Credit_Spread'], scores['Yield_Curve'] = 50, 50

    # ---- 模組二：市場情緒與衍生商品 (權重 20%) ----
    try:
        vxn = yf.Ticker('^VXN').history(period='5d')['Close'].iloc[-1]
        scores['Market_Panic'] = scale_score(vxn, 15, 35)
    except:
        scores['Market_Panic'] = 50

    # ---- 模組三：微觀估值與現金流 (權重 30%) ----
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

    # ---- 模組四：亞洲供應鏈高頻動能 (權重 20%) ----
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

    # ---- 計算複合風險總分 (Composite Risk Score) ----
    total_score = (
        (scores['Credit_Spread'] * 0.15 + scores['Yield_Curve'] * 0.15) +  # 總經 30%
        (scores['Market_Panic'] * 0.20) +                                  # 情緒 20%
        (scores['Valuation'] * 0.20 + scores['FCF_Warning'] * 0.10) +      # 估值 30%
        (scores['Supply_Chain'] * 0.20)                                    # 供應鏈 20%
    )

    # ---- 取得燈號與訊息 ----
    zone_label, status_message = get_status_color_and_message(total_score)

    # ---- 輸出報告 ----
    print(f"\n【📊 AI 泡沫複合風險指數】: {total_score:.2f} / 100")
    print(f"燈號狀態: {zone_label}")
    print(f"行動建議: {status_message}")
    print("\n分項指標風險評分 (分數越高越危險，最高100分):")
    print(f" 1. 美國信用利差風險: {scores['Credit_Spread']:.1f}")
    print(f" 2. 殖利率曲線陡峭化風險: {scores['Yield_Curve']:.1f}")
    print(f" 3. 選擇權與衍生商品恐慌度: {scores['Market_Panic']:.1f}")
    print(f" 4. 科技巨頭核心估值倍數: {scores['Valuation']:.1f} (平均 EV/EBITDA: {avg_ev_ebitda:.1f}x)")
    print(f" 5. 供應鏈(台積電等ADR)動能風險: {scores['Supply_Chain']:.1f}")
    print("==================================================")

if __name__ == "__main__":
    run_quant_model()