# Tradingview-strategy
This repo contains Pine Script strategies for TradingView.

## Getting started

1. Install Python 3.11+ and create a virtual environment if desired.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file (or export environment variables) with at least:

   ```bash
   BOT_INITIAL_EQUITY=1000   # your starting USDT balance
   TG_BOT_TOKEN=             # optional: Telegram bot token
   TG_CHAT_ID=               # optional: chat/group ID for alerts
   ```

4. Run the live strategy script:

   ```bash
   python ETH_BTC_MTF_LIVE.py
   ```

## Multi-Market Dominance EMA Break Strategy

`multi_market_ema_strategy.pine` monitors BTC dominance, USDT dominance, and TOTAL3 using configurable 5/8/13 EMAs on a daily timeframe. The strategy issues longs when BTC.D and TOTAL3 experience bullish EMA stack breaks **while** USDT.D simultaneously experiences a bearish stack break; shorts trigger when BTC.D and TOTAL3 flip bearish and USDT.D flips bullish. It also keeps track of the chart's daily volume and highlights cases where volume grows between +30% and +70%, generating an alert and an on-chart label.
