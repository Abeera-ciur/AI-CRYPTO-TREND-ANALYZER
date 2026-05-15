
🚀 CRYPTO TRADING AI — Signal Generator v2.0

<img width="1600" height="894" alt="Visual Dashboard- Can also view from files" src="https://github.com/user-attachments/assets/ffe8e318-67ef-4842-89df-204bcd7d37e8" />

📖 Overview
This is a prototype
 cryptocurrency trading signal generator built with Python and Streamlit. It analyzes multiple technical indicators using a vote-based system to produce BUY, SELL, or HOLD signals with confidence scores and risk management levels (entry, stop-loss, take-profit).

✨ Key Features
Feature	Description
Vote-based signals	Each indicator casts weighted BULL/BEAR votes (no canceling out)
8+ indicators	EMA, RSI, MACD, Bollinger Bands, Stochastic, Volume, ATR, Candle patterns
Multi-pair scanner	Scan up to 6 pairs simultaneously
Interactive charts	Plotly candlestick charts with indicators
Auto-refresh	Real-time updates (60s default, configurable)
AI commentary	Optional NVIDIA Llama 3.1 70B analysis
📊 Technical Indicators Used
Indicator	Weight	What it detects
EMA Stack (9/20/50)	3 pts	Trend alignment
Golden/Death Cross	5 pts	Major trend reversal
RSI (overbought/oversold)	4 pts	Momentum extremes
MACD crossover	4 pts	Momentum shift
Bollinger Bands	4 pts	Volatility extremes
Stochastic RSI	3 pts	Oversold/overbought
Volume surge	2 pts	Confirmation
Price vs EMA200	2 pts	Long-term bias
Threshold: Signal triggers when BULL OR BEAR score ≥ 5 points (configurable: 3-12)

🚀 Quick Start
1. Install dependencies
bash
pip install -r requirements.txt
2. Run the app
bash
streamlit run ai-crypto.py
3. Open browser
Navigate to http://localhost:8501

⚙️ Configuration
API Keys (Optional)
python
# In ai-crypto.py
NVIDIA_API_KEY   = "nvapi-..."  # Get from https://build.nvidia.com
BINANCE_API_KEY  = ""           # Optional (public data works without)
BINANCE_API_SECRET = ""
Adjustable Settings (in UI)
Setting	Range	Default
Signal threshold	3-12 pts	5 pts
Auto-refresh	15-300 sec	60 sec
Timeframes	5m, 15m, 1h, 4h, 1D	1h
Pairs	BTC, ETH, SOL, BNB, XRP, DOGE	BTCUSDT
📁 File Structure
text
.
├── ai-crypto.py          # Main application
├── requirements.txt      # Python dependencies
└── README.md            # This file
🧠 How It Works
Signal Generation Flow
text
Market Data (Binance)
        ↓
OHLCV DataFrame (300 candles)
        ↓
Indicators.add_all() → EMAs, RSI, MACD, BB, Stoch, ATR, Volume
        ↓
SignalEngine.generate() → Vote tallying
        ↓
Output: BUY/SELL/HOLD + Entry/SL/TP + Confidence%
Vote Example (BUY Signal)
text
✅ EMA stack bullish (9>20>50)      +3
✅ Price above EMA20                +1
✅ RSI recovering from oversold     +3
✅ MACD histogram flipped positive  +4
✅ Price in lower Bollinger zone    +2
─────────────────────────────────────
🐂 BULL TOTAL: 13 pts  |  🐻 BEAR: 2 pts
→ BUY SIGNAL (86% confidence)
Risk Management
Signal	Entry	Stop Loss (2× ATR)	Take Profit (4× ATR)
BUY	Current price	Price - 2×ATR	Price + 4×ATR
SELL	Current price	Price + 2×ATR	Price - 4×ATR
📊 Output Examples
Signal Card
text
┌─────────────────────────────┐
│      🟢 BUY SIGNAL          │
│  🎯 Entry:    $42,850.00    │
│  🛑 Stop Loss: $41,200.00   │
│  💰 Take Profit: $46,150.00 │
│  📊 Confidence: 78%         │
│  ⚖️ Risk:Reward: 1:2.5     │
└─────────────────────────────┘
Console Logging
text
═══════════════════════════════════════════════════════
  BTCUSDT  [1h]   BUY   conf=78%   bull=13  bear=2
  Price: 42850.00   ATR: 825.00
  Entry=42850  SL=41200  TP=46150  RR=2.5
    ✅ EMA stack bullish (9>20>50)
    ✅ MACD histogram flipped positive
    ✅ RSI recovering from oversold (34.2 ↑)
═══════════════════════════════════════════════════════
🔧 Troubleshooting
Issue	Solution
No module named 'streamlit'	Run pip install streamlit
Binance connection error	Check internet or increase timeout
Empty dataframe	Reduce limit or check symbol format (e.g., "BTCUSDT")
AI commentary not working	Verify NVIDIA API key or disable
⚠️ Disclaimer
text
⚠️ EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE

This software is for learning purposes. Cryptocurrency trading 
involves substantial risk of loss. Always do your own research 
and never trade with money you cannot afford to lose.
📈 Performance Tips
Default threshold (5) works well for 1h timeframe

Lower threshold (3-4) → More signals, higher noise

Higher threshold (7-10) → Fewer signals, higher confidence

Shorter timeframes (5m-15m) → Increase threshold to reduce false signals

Longer timeframes (4h-1D) → Lower threshold works fine

🔄 Updating Indicators
To add custom indicators, modify Indicators.add_all() and add corresponding voting logic in SignalEngine.generate().

Built with: Python, Streamlit, CCXT, TA-Lib wrapper, Plotly
