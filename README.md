# AI Bubble Monitor v7.0

**Global Macro Regime Radar** — 15-Factor Weighted Stress Scoring System for AI Bubble Risk Assessment

[![Hourly Monitor](https://github.com/YenLin1988/ai-bubble-monitor/actions/workflows/hourly_monitor.yml/badge.svg)](https://github.com/YenLin1988/ai-bubble-monitor/actions/workflows/hourly_monitor.yml)

---

## What is this?

A real-time dashboard that monitors **whether the AI bubble is at risk of bursting** by combining 15 macroeconomic, financial, and sentiment indicators into a single weighted stress score (0-100).

The system classifies the current market regime into one of four states:

| Regime | Condition | Meaning |
|--------|-----------|---------|
| Expansion | Growth > 0 & Stress < 50 | Healthy growth, stay long |
| Overheating | Growth > 0 & Stress >= 50 | Late cycle, start hedging |
| Bust | Growth < 0 & Stress >= 50 | Bubble bursting, risk-off |
| Contraction | Growth < 0 & Stress < 50 | Bottom forming, accumulate |

---

## Dashboard Preview

### Regime Banner & KPI Cards
The top section shows the current regime classification, weighted stress score, and 15 real-time macro indicators across KPI cards.

### 15-Factor Risk Radar
A polar chart decomposing risk across all 15 dimensions, with a safe-zone overlay at score 30.

### Cluster Stress Decomposition
Four cluster cards showing weighted contributions from each risk group, with per-factor drill-down.

### AI CapEx Tracker
Quarterly capital expenditure for MSFT, GOOG, AMZN, and META, plus CapEx-as-%-of-revenue trend with a 25% danger threshold.

---

## 15-Factor Scoring Model

### Y-Axis: Weighted Stress Score (0-100)

Organized into 4 clusters:

#### Cluster A: Monetary Policy & Liquidity (33%)

| Factor | Source | Safe | Danger | Weight |
|--------|--------|------|--------|--------|
| 10Y Treasury Yield | FRED: GS10 / Yahoo: ^TNX | 3.5% | 5.0% | 7% |
| Fed Funds Rate | FRED: FEDFUNDS | 3.0% | 5.5% | 7% |
| Net Liquidity (Fed B/S - TGA - RRP) | FRED: WALCL, WTREGEN, RRPONTSYD | $6.2T | $5.4T | 7% |
| M2 Money Supply YoY | FRED: M2SL | +6% | -2% | 5% |
| JPY Liquidity Stress | Yahoo: JPY=X (USD/JPY + 20d momentum) | 150+ (stable) | 130 (liquidity crunch) | 7% |

#### Cluster B: Credit & Financial Conditions (19%)

| Factor | Source | Safe | Danger | Weight |
|--------|--------|------|--------|--------|
| 10Y-2Y Yield Curve | FRED: T10Y2Y | -0.8% | +0.2% | 7% |
| HY Credit Spread | FRED: BAMLH0A0HYM2 | 3.0% | 5.5% | 6% |
| Chicago Fed NFCI | FRED: NFCI | -0.5 | +0.5 | 6% |

#### Cluster C: Valuation & Sentiment (24%)

| Factor | Source | Safe | Danger | Weight |
|--------|--------|------|--------|--------|
| AI Sector EV/EBITDA | Yahoo Finance (NVDA, MSFT, META, GOOG, AMZN) | 25x | 45x | 7% |
| Buffett Indicator | FRED: WILL5000PR / GDP | 100% | 200% | 6% |
| VXN (Nasdaq Volatility) | Yahoo: ^VXN | 15 | 35 | 5% |
| U. Michigan Consumer Sentiment | FRED: UMCSENT | 70+ | <30 | 6% |

#### Cluster D: Macro & Inflation (24%)

| Factor | Source | Safe | Danger | Weight |
|--------|--------|------|--------|--------|
| US Dollar Index | Yahoo: DX-Y.NYB | 100 | 107 | 8% |
| Commodity Momentum (20d) | Yahoo: CL=F, GC=F | 0% | +10% | 8% |
| Unemployment Rate | FRED: UNRATE | 3.5% | 6.0% | 8% |

### X-Axis: AI Growth Momentum

| Component | Ticker(s) | Weight |
|-----------|-----------|--------|
| Taiwan Supply Chain | TSM, UMC, ASX | 20% |
| AI Infrastructure | MU, VRT | 15% |
| Semiconductor Equipment | ASML, AMAT, LRCX | 15% |
| AI Networking Chips | AVGO | 10% |
| SOX Semiconductor Index | ^SOX | 25% |
| Bitcoin (Risk Appetite Proxy) | BTC-USD | 15% |

---

## Additional Data Tracked (Not in Scoring)

| Indicator | Purpose |
|-----------|---------|
| FINRA Margin Debt | Leverage / speculation level |
| CBOE Put/Call Ratio | Options market fear gauge |
| VIX Fear Index | Market fear fallback when P/C unavailable |
| Hyperscaler CapEx (MSFT/GOOG/AMZN/META) | AI arms race spending intensity |
| Per-stock valuation matrix (P/E, P/S, P/FCF, PEG) | Individual AI stock health |
| Historical stress score archive | Trend analysis over time |
| Δ Stress velocity (24H / 7D / 30D) | Rate of stress change — speed matters |
| Historical crisis overlay (2000 / 2008 / 2022) | Pattern comparison with past bubbles |

---

## Conditional Alert System

The dashboard automatically triggers warnings when thresholds are breached. Optional external dispatch supports Email, Discord, and generic JSON webhooks. Telegram push is intentionally not included.

| Condition | Alert |
|-----------|-------|
| Consumer Sentiment < 55 | Recession early warning |
| NFCI > 0.2 | Financial conditions tightening |
| M2 YoY < -2% | Liquidity contraction (rare, high severity) |
| Buffett Indicator > 200% | Extreme market overvaluation |
| Stress Score >= 65 / 75 | Elevated / danger stress zone |
| 24H Stress Delta >= 5 | Fast stress acceleration |
| Bust regime | Negative AI momentum plus elevated macro stress |

External dispatch is disabled by default. Copy `.env.example`, set the channel credentials, and set `ALERTS_ENABLED=1`.

---

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) with custom institutional dark theme
- **Charts**: [Plotly](https://plotly.com/python/) with Bloomberg-style color palette
- **Data**: [FRED](https://fred.stlouisfed.org/) (macroeconomic) + [Yahoo Finance](https://finance.yahoo.com/) (market data)
- **Automation**: GitHub Actions (hourly cron)
- **Storage**: SQLite for historical stress score archiving, with legacy JSON migration

---

## Quick Start

### Local Development

```bash
# Clone
git clone https://github.com/YenLin1988/ai-bubble-monitor.git
cd ai-bubble-monitor

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run Ai_bubble_monitor.py
```

The dashboard will open at `http://localhost:8501`.

### Deploy to Streamlit Cloud

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select `Ai_bubble_monitor.py` as the main file
5. Deploy

---

## Project Structure

```
ai-bubble-monitor/
  Ai_bubble_monitor.py          # Main Streamlit application
  bubble_monitor/               # Scoring, config, data quality, storage, alerts
  docs/
    ARCHITECTURE.md             # Runtime architecture and alert policy
  tests/                        # Focused unit tests for non-UI core logic
  requirements.txt              # Python dependencies
  .github/workflows/
    hourly_monitor.yml           # GitHub Actions hourly cron job
  .gitignore                    # Excludes local history file
  data/stress_history.sqlite3   # Auto-generated SQLite history (gitignored)
  README.md                    # This file
```

---

## Design Philosophy

This tool is built on the premise that **AI bubble risk cannot be measured by a single indicator**. Instead, it synthesizes signals across four domains:

1. **Is money getting more expensive?** (Monetary cluster)
2. **Is credit drying up?** (Credit cluster)
3. **Are valuations stretched beyond fundamentals?** (Valuation cluster)
4. **Is the real economy weakening?** (Macro cluster)

When all four clusters are elevated simultaneously while AI sector growth momentum turns negative, the system signals a **Bust** regime.

The scoring model uses a linear normalization between empirically calibrated "safe" and "danger" thresholds for each factor, then applies cluster-weighted averaging to produce the final stress score.

---

## Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Market data is delayed 15-20 minutes. Always consult a qualified financial advisor before making investment decisions.

---

## License

MIT

---

Built with Streamlit, Plotly, and data from FRED & Yahoo Finance.
