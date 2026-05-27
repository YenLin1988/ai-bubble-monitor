"""Model configuration for the AI Bubble Monitor scoring system."""

from __future__ import annotations

TECH_GIANTS = ["NVDA", "MSFT", "META", "GOOG", "AMZN"]
TW_SUPPLY = ["TSM", "UMC", "ASX"]
INFRA_STOCKS = ["MU", "VRT"]
EQUIP_STOCKS = ["ASML", "AMAT", "LRCX"]
NETWORK_STOCKS = ["AVGO"]
COMMODITIES = ["CL=F", "GC=F"]
HYPERSCALERS = ["MSFT", "GOOG", "AMZN", "META"]

FRED_SERIES = {
    "spread": "BAMLH0A0HYM2",
    "curve": "T10Y2Y",
    "walcl": "WALCL",
    "tga": "WTREGEN",
    "rrp": "RRPONTSYD",
    "margin_debt": "BOGZ1FL663067003Q",
    "wilshire": "WILL5000PR",
    "gdp": "GDP",
    "fedfunds": "FEDFUNDS",
    "m2": "M2SL",
    "nfci": "NFCI",
    "unrate": "UNRATE",
    "umcsent": "UMCSENT",
}

RISK_LABELS_ZH = {
    "10Y Treasury": "定價引力",
    "Fed Funds Rate": "聯準會利率",
    "Net Liquidity": "絕對資金",
    "M2 Supply YoY": "貨幣供給",
    "JPY Liquidity": "日圓流動性",
    "Yield Curve": "衰退警報",
    "Credit Spread": "違約風險",
    "NFCI": "金融環境",
    "EV/EBITDA": "估值極限",
    "Buffett Indicator": "巴菲特指標",
    "VXN Volatility": "市場情緒",
    "Consumer Sentiment": "消費者信心",
    "US Dollar": "全球水閘",
    "Commodities": "通膨預期",
    "Unemployment": "就業惡化",
}

RISK_CLUSTERS = {
    "10Y Treasury": "A 貨幣流動性",
    "Fed Funds Rate": "A 貨幣流動性",
    "Net Liquidity": "A 貨幣流動性",
    "M2 Supply YoY": "A 貨幣流動性",
    "JPY Liquidity": "A 貨幣流動性",
    "Yield Curve": "B 信用條件",
    "Credit Spread": "B 信用條件",
    "NFCI": "B 信用條件",
    "EV/EBITDA": "C 估值情緒",
    "Buffett Indicator": "C 估值情緒",
    "VXN Volatility": "C 估值情緒",
    "Consumer Sentiment": "C 估值情緒",
    "US Dollar": "D 總體通膨",
    "Commodities": "D 總體通膨",
    "Unemployment": "D 總體通膨",
}

WEIGHTS = {
    "10Y Treasury": 0.07,
    "Fed Funds Rate": 0.07,
    "Net Liquidity": 0.07,
    "M2 Supply YoY": 0.05,
    "JPY Liquidity": 0.07,
    "Yield Curve": 0.07,
    "Credit Spread": 0.06,
    "NFCI": 0.06,
    "EV/EBITDA": 0.07,
    "Buffett Indicator": 0.06,
    "VXN Volatility": 0.05,
    "Consumer Sentiment": 0.06,
    "US Dollar": 0.08,
    "Commodities": 0.08,
    "Unemployment": 0.08,
}

FACTOR_METHODOLOGY = [
    {
        "factor": "10Y Treasury",
        "source": "Yahoo Finance ^TNX",
        "safe": "3.5%",
        "danger": "5.0%",
        "weight": "7%",
        "cluster": RISK_CLUSTERS["10Y Treasury"],
    },
    {
        "factor": "Fed Funds Rate",
        "source": "FRED FEDFUNDS",
        "safe": "3.0%",
        "danger": "5.5%",
        "weight": "7%",
        "cluster": RISK_CLUSTERS["Fed Funds Rate"],
    },
    {
        "factor": "Net Liquidity",
        "source": "FRED WALCL - WTREGEN - RRPONTSYD",
        "safe": "$6.2T",
        "danger": "$5.4T",
        "weight": "7%",
        "cluster": RISK_CLUSTERS["Net Liquidity"],
    },
    {
        "factor": "M2 Supply YoY",
        "source": "FRED M2SL",
        "safe": "+6%",
        "danger": "-2%",
        "weight": "5%",
        "cluster": RISK_CLUSTERS["M2 Supply YoY"],
    },
    {
        "factor": "JPY Liquidity",
        "source": "Yahoo Finance JPY=X",
        "safe": "USD/JPY 150+ stable",
        "danger": "USD/JPY 130 or sharp yen rally",
        "weight": "7%",
        "cluster": RISK_CLUSTERS["JPY Liquidity"],
    },
    {
        "factor": "Yield Curve",
        "source": "FRED T10Y2Y",
        "safe": "-0.8%",
        "danger": "+0.2%",
        "weight": "7%",
        "cluster": RISK_CLUSTERS["Yield Curve"],
    },
    {
        "factor": "Credit Spread",
        "source": "FRED BAMLH0A0HYM2",
        "safe": "3.0%",
        "danger": "5.5%",
        "weight": "6%",
        "cluster": RISK_CLUSTERS["Credit Spread"],
    },
    {
        "factor": "NFCI",
        "source": "FRED NFCI",
        "safe": "-0.5",
        "danger": "+0.5",
        "weight": "6%",
        "cluster": RISK_CLUSTERS["NFCI"],
    },
    {
        "factor": "EV/EBITDA",
        "source": "Yahoo Finance AI mega-cap valuation",
        "safe": "25x",
        "danger": "45x",
        "weight": "7%",
        "cluster": RISK_CLUSTERS["EV/EBITDA"],
    },
    {
        "factor": "Buffett Indicator",
        "source": "FRED WILL5000PR / GDP",
        "safe": "100%",
        "danger": "200%",
        "weight": "6%",
        "cluster": RISK_CLUSTERS["Buffett Indicator"],
    },
    {
        "factor": "VXN Volatility",
        "source": "Yahoo Finance ^VXN",
        "safe": "15",
        "danger": "35",
        "weight": "5%",
        "cluster": RISK_CLUSTERS["VXN Volatility"],
    },
    {
        "factor": "Consumer Sentiment",
        "source": "FRED UMCSENT",
        "safe": "70+",
        "danger": "<30",
        "weight": "6%",
        "cluster": RISK_CLUSTERS["Consumer Sentiment"],
    },
    {
        "factor": "US Dollar",
        "source": "Yahoo Finance DX-Y.NYB",
        "safe": "100",
        "danger": "107",
        "weight": "8%",
        "cluster": RISK_CLUSTERS["US Dollar"],
    },
    {
        "factor": "Commodities",
        "source": "Yahoo Finance CL=F, GC=F 20D momentum",
        "safe": "0%",
        "danger": "+10%",
        "weight": "8%",
        "cluster": RISK_CLUSTERS["Commodities"],
    },
    {
        "factor": "Unemployment",
        "source": "FRED UNRATE",
        "safe": "3.5%",
        "danger": "6.0%",
        "weight": "8%",
        "cluster": RISK_CLUSTERS["Unemployment"],
    },
]

GROWTH_COMPONENTS = [
    {"name": "Taiwan Supply Chain", "tickers": "TSM, UMC, ASX", "weight": "20%"},
    {"name": "AI Infrastructure", "tickers": "MU, VRT", "weight": "15%"},
    {"name": "Semiconductor Equipment", "tickers": "ASML, AMAT, LRCX", "weight": "15%"},
    {"name": "AI Networking Chips", "tickers": "AVGO", "weight": "10%"},
    {"name": "SOX Semiconductor Index", "tickers": "^SOX", "weight": "25%"},
    {"name": "Bitcoin Risk Appetite", "tickers": "BTC-USD", "weight": "15%"},
]
