import time
import math
import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv

# ==========================
# بارگذاری .env
# ==========================

load_dotenv(dotenv_path=".env")

# ==========================
# تنظیمات کلی
# ==========================

BASE_URL = "https://api.binance.com"

WARMUP_BARS = 3          # چند کندل آخر را هر بار دوباره چک کند
POLL_INTERVAL = 60       # فاصله بین هر حلقه (ثانیه)
ATR_LEN = 14

# مدیریت سرمایه (اکوییتی مشترک برای همه‌ی تریدها روی BTC و ETH)
equity_env = os.getenv("BOT_INITIAL_EQUITY")
if not equity_env:
    raise ValueError("BOT_INITIAL_EQUITY is not set in the environment/.env file.")

try:
    INITIAL_EQUITY = float(equity_env)
except ValueError as exc:
    raise ValueError("BOT_INITIAL_EQUITY must be a numeric value.") from exc
RISK_PCT = 1.0           # درصد ریسک هر ترید از equity (مثلاً 1%)

# ==========================
# Telegram config
# ==========================

TELEGRAM_ENABLED   = True
TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TG_CHAT_ID", "")


def send_telegram(text: str):
    if not TELEGRAM_ENABLED:
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] Skipped: TG_BOT_TOKEN / TG_CHAT_ID not set (check .env).")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print("[TG] ERROR:", r.status_code, r.text)
        else:
            print("[TG] Sent.")
    except Exception as e:
        print("[TG] EXCEPTION:", e)


def notify_signal(text: str):
    print(text)
    send_telegram(text)


def summarize_combos(strategies: list) -> dict:
    """Count how many strategy setups exist per (symbol, timeframe) combo."""
    combos: dict = {}
    for strat in strategies:
        key = (strat["symbol"], strat["tf"])
        combos[key] = combos.get(key, 0) + 1
    return combos

# ==========================
# ابزارهای دیتا و اندیکاتور
# ==========================

def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for row in data:
        rows.append({
            "open_time": datetime.fromtimestamp(row[0] / 1000.0, tz=timezone.utc),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def compute_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=length, adjust=False).mean()
    return atr


def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    macd_fast   = params["macd_fast"]
    macd_slow   = params["macd_slow"]
    macd_signal = params["macd_signal"]
    rsi_len     = params["rsi_len"]
    ema_len     = params["ema_trend_len"]

    close = df["close"]

    # MACD
    fastMA = close.ewm(span=macd_fast, adjust=False).mean()
    slowMA = close.ewm(span=macd_slow, adjust=False).mean()
    macd   = fastMA - slowMA
    signal = macd.ewm(span=macd_signal, adjust=False).mean()

    # RSI
    rsi = compute_rsi(close, rsi_len)

    # EMA Trend
    emaT = close.ewm(span=ema_len, adjust=False).mean()

    # ATR
    atr = compute_atr(df, ATR_LEN)

    df = df.copy()
    df["macd"]   = macd
    df["signal"] = signal
    df["rsi"]    = rsi
    df["emaT"]   = emaT
    df["atr"]    = atr

    df = df.dropna().reset_index(drop=True)
    return df

# ==========================
# مدیریت پوزیشن‌ها (Paper Trading)
# ==========================

class Position:
    def __init__(self, symbol, tf, strategy_name, direction, entry, sl, tp, qty, open_time):
        self.symbol = symbol
        self.tf = tf
        self.strategy_name = strategy_name
        self.direction = direction  # "LONG" or "SHORT"
        self.entry = entry
        self.sl = sl
        self.tp = tp
        self.qty = qty
        self.open_time = open_time
        self.status = "OPEN"
        self.close_time = None
        self.close_price = None
        self.pnl = 0.0

    def __repr__(self):
        return f"<Pos {self.symbol} {self.tf} {self.direction} entry={self.entry} sl={self.sl} tp={self.tp} qty={self.qty} status={self.status}>"


def process_exits(symbol: str, tf: str, df: pd.DataFrame, positions: list, equity: float):
    """
    روی تمام پوزیشن‌های باز در این نماد/تایم‌فریم می‌گردد و اگر high/low کندل SL/TP را زد، پوزیشن را می‌بندد.
    """
    for idx in range(len(df)):
        bar_time = df.loc[idx, "open_time"]
        high = df.loc[idx, "high"]
        low = df.loc[idx, "low"]

        for pos in positions:
            if pos.symbol != symbol or pos.tf != tf or pos.status != "OPEN":
                continue
            # فقط کندل‌هایی بعد از زمان باز شدن را چک کن
            if bar_time <= pos.open_time:
                continue

            closed = False
            close_price = None
            reason = None

            if pos.direction == "LONG":
                # اول SL، بعد TP (محافظه‌کارانه)
                if low <= pos.sl:
                    close_price = pos.sl
                    reason = "SL"
                    closed = True
                elif high >= pos.tp:
                    close_price = pos.tp
                    reason = "TP"
                    closed = True
            else:  # SHORT
                if high >= pos.sl:
                    close_price = pos.sl
                    reason = "SL"
                    closed = True
                elif low <= pos.tp:
                    close_price = pos.tp
                    reason = "TP"
                    closed = True

            if closed and close_price is not None:
                pos.status = "CLOSED"
                pos.close_time = bar_time
                pos.close_price = close_price

                if pos.direction == "LONG":
                    pos.pnl = (close_price - pos.entry) * pos.qty
                else:
                    pos.pnl = (pos.entry - close_price) * pos.qty

                equity += pos.pnl

                text = (
                    f"*TRADE CLOSED* `{symbol}` `{tf}` {reason}\n"
                    f"Strategy: `{pos.strategy_name}`\n"
                    f"Direction: {pos.direction}\n"
                    f"Entry: `{pos.entry:.2f}`  Close: `{close_price:.2f}`\n"
                    f"Qty: `{pos.qty:.4f}`  PnL: `{pos.pnl:.2f} USDT`\n"
                    f"New Equity: `{equity:.2f} USDT`\n"
                    f"Open: {pos.open_time}  Close: {pos.close_time}"
                )
                notify_signal(text)

    return equity

# ==========================
# تولید سیگنال + ورود پوزیشن
# ==========================

def generate_signals_and_trades(symbol: str,
                                df: pd.DataFrame,
                                params: dict,
                                last_idx: int,
                                positions: list,
                                equity: float):
    tf = params["tf"]
    rsi_bull = params["rsi_bull"]
    rsi_bear = params["rsi_bear"]
    strat_name = params.get("name", f"{symbol}_{tf}")

    if len(df) < 3:
        return last_idx, positions, equity

    # اول خروج‌ها را روی کل دیتافریم اعمال کن
    equity = process_exits(symbol, tf, df, positions, equity)

    start_idx = max(last_idx, len(df) - WARMUP_BARS - 1)
    end_idx = len(df) - 1
    if start_idx < 1:
        start_idx = 1

    for i in range(start_idx, end_idx + 1):
        c = df.loc[i, "close"]
        macd_curr = df.loc[i, "macd"]
        macd_prev = df.loc[i - 1, "macd"]
        sig_curr  = df.loc[i, "signal"]
        sig_prev  = df.loc[i - 1, "signal"]
        rsi_curr  = df.loc[i, "rsi"]
        rsi_prev  = df.loc[i - 1, "rsi"]
        ema_curr  = df.loc[i, "emaT"]
        atr_curr  = df.loc[i, "atr"]
        t = df.loc[i, "open_time"]

        trend_up   = c > ema_curr
        trend_down = c < ema_curr

        bull_macd_cross = (macd_prev <= sig_prev) and (macd_curr > sig_curr) and (macd_prev < 0)
        bear_macd_cross = (macd_prev >= sig_prev) and (macd_curr < sig_curr) and (macd_prev > 0)

        rsi_bull_cross = (rsi_prev <= rsi_bull) and (rsi_curr > rsi_bull)
        rsi_bear_cross = (rsi_prev >= rsi_bear) and (rsi_curr < rsi_bear)

        long_signal  = trend_up and bull_macd_cross and rsi_bull_cross
        short_signal = trend_down and bear_macd_cross and rsi_bear_cross

        if not math.isfinite(atr_curr) or atr_curr <= 0:
            continue

        # اگر همین الان در این نماد+TF پوزیشن باز داریم، پوزیشن جدید نده
        has_open = any(
            p.symbol == symbol and p.tf == tf and p.status == "OPEN"
            for p in positions
        )
        if has_open:
            continue

        # محاسبه سایز پوزیشن بر اساس ریسک درصدی
        risk_value = equity * (RISK_PCT / 100.0)

        if long_signal:
            entry = c
            sl = entry - atr_curr
            tp = entry + 2.0 * atr_curr
            stop_dist = max(abs(entry - sl), 1e-6)
            qty = risk_value / stop_dist

            if qty > 0:
                pos = Position(symbol, tf, strat_name, "LONG", entry, sl, tp, qty, t)
                positions.append(pos)
                text = (
                    f"*NEW TRADE* `{symbol}` `{tf}` LONG\n"
                    f"Strategy: `{strat_name}`\n"
                    f"Entry: `{entry:.2f}` SL: `{sl:.2f}` TP: `{tp:.2f}`\n"
                    f"Qty: `{qty:.4f}`  Risk: `{risk_value:.2f} USDT`  Equity: `{equity:.2f} USDT`\n"
                    f"(MACD up + RSI>{rsi_bull} + Trend Up)"
                )
                notify_signal(text)

        if short_signal:
            entry = c
            sl = entry + atr_curr
            tp = entry - 2.0 * atr_curr
            stop_dist = max(abs(entry - sl), 1e-6)
            qty = risk_value / stop_dist

            if qty > 0:
                pos = Position(symbol, tf, strat_name, "SHORT", entry, sl, tp, qty, t)
                positions.append(pos)
                text = (
                    f"*NEW TRADE* `{symbol}` `{tf}` SHORT\n"
                    f"Strategy: `{strat_name}`\n"
                    f"Entry: `{entry:.2f}` SL: `{sl:.2f}` TP: `{tp:.2f}`\n"
                    f"Qty: `{qty:.4f}`  Risk: `{risk_value:.2f} USDT`  Equity: `{equity:.2f} USDT`\n"
                    f"(MACD down + RSI<{rsi_bear} + Trend Down)"
                )
                notify_signal(text)

    return end_idx, positions, equity

# ==========================
# تعریف استراتژی‌ها برای BTC و ETH
# ==========================

STRATEGIES = [
    # ---------------- BTCUSDT (همان استراتژی قبلی بیت‌کوین) ----------------
    {
        "symbol": "BTCUSDT",
        "name": "BTC_15m",
        "tf": "15m",
        "macd_fast": 8,
        "macd_slow": 31,
        "macd_signal": 4,
        "rsi_len": 10,
        "rsi_bull": 55,
        "rsi_bear": 70,
        "ema_trend_len": 200,
    },
    {
        "symbol": "BTCUSDT",
        "name": "BTC_1h",
        "tf": "1h",
        "macd_fast": 10,
        "macd_slow": 26,
        "macd_signal": 6,
        "rsi_len": 14,
        "rsi_bull": 50,
        "rsi_bear": 70,
        "ema_trend_len": 150,
    },
    {
        "symbol": "BTCUSDT",
        "name": "BTC_4h",
        "tf": "4h",
        "macd_fast": 13,
        "macd_slow": 31,
        "macd_signal": 9,
        "rsi_len": 20,
        "rsi_bull": 45,
        "rsi_bear": 60,
        "ema_trend_len": 150,
    },
    {
        "symbol": "BTCUSDT",
        "name": "BTC_15m",
        "tf": "15m",
        "macd_fast": 8,
        "macd_slow": 31,
        "macd_signal": 4,
        "rsi_len": 10,
        "rsi_bull": 55,
        "rsi_bear": 70,
        "ema_trend_len": 200,
    },
    {
        "symbol": "BTCUSDT",
        "name": "BTC_1h",
        "tf": "1h",
        "macd_fast": 10,
        "macd_slow": 26,
        "macd_signal": 6,
        "rsi_len": 14,
        "rsi_bull": 50,
        "rsi_bear": 70,
        "ema_trend_len": 150,
    },
    {
        "symbol": "BTCUSDT",
        "name": "BTC_4h",
        "tf": "4h",
        "macd_fast": 13,
        "macd_slow": 31,
        "macd_signal": 9,
        "rsi_len": 20,
        "rsi_bull": 45,
        "rsi_bear": 60,
        "ema_trend_len": 150,
    },

    # ----- BTCUSDT Daily از فایل 1d جدید -----
    {
        "symbol": "BTCUSDT",
        "name": "BTC_1d_1",
        "tf": "1d",
        "macd_fast": 10,
        "macd_slow": 22,
        "macd_signal": 4,
        "rsi_len": 12,
        "rsi_bull": 45,
        "rsi_bear": 55,
        "ema_trend_len": 100,
    },
    {
        "symbol": "BTCUSDT",
        "name": "BTC_1d_2",
        "tf": "1d",
        "macd_fast": 10,
        "macd_slow": 31,
        "macd_signal": 4,
        "rsi_len": 18,
        "rsi_bull": 50,
        "rsi_bear": 55,
        "ema_trend_len": 100,
    },
    # ---------------- ETHUSDT – ۲ استراتژی برتر در هر TF ----------------
    # 1D – دو استراتژی برتر از فایل 1d
    {
        "symbol": "ETHUSDT",
        "name": "ETH_1d_1",
        "tf": "1d",
        "macd_fast": 12,
        "macd_slow": 16,
        "macd_signal": 4,
        "rsi_len": 14,
        "rsi_bull": 40,
        "rsi_bear": 65,
        "ema_trend_len": 50,
    },
    {
        "symbol": "ETHUSDT",
        "name": "ETH_1d_2",
        "tf": "1d",
        "macd_fast": 6,
        "macd_slow": 16,
        "macd_signal": 8,
        "rsi_len": 14,
        "rsi_bull": 40,
        "rsi_bear": 65,
        "ema_trend_len": 100,
    },

    # 4H – دو استراتژی برتر از فایل 4h
    {
        "symbol": "ETHUSDT",
        "name": "ETH_4h_1",
        "tf": "4h",
        "macd_fast": 10,
        "macd_slow": 31,
        "macd_signal": 6,
        "rsi_len": 10,
        "rsi_bull": 50,
        "rsi_bear": 60,
        "ema_trend_len": 75,
    },
    {
        "symbol": "ETHUSDT",
        "name": "ETH_4h_2",
        "tf": "4h",
        "macd_fast": 10,
        "macd_slow": 31,
        "macd_signal": 6,
        "rsi_len": 10,
        "rsi_bull": 55,
        "rsi_bear": 60,
        "ema_trend_len": 75,
    },

    # 1H – دو استراتژی برتر از فایل 1h
    {
        "symbol": "ETHUSDT",
        "name": "ETH_1h_1",
        "tf": "1h",
        "macd_fast": 10,
        "macd_slow": 19,
        "macd_signal": 6,
        "rsi_len": 10,
        "rsi_bull": 45,
        "rsi_bear": 60,
        "ema_trend_len": 50,
    },
    {
        "symbol": "ETHUSDT",
        "name": "ETH_1h_2",
        "tf": "1h",
        "macd_fast": 6,
        "macd_slow": 19,
        "macd_signal": 10,
        "rsi_len": 10,
        "rsi_bull": 45,
        "rsi_bear": 65,
        "ema_trend_len": 50,
    },

    # 15M – دو استراتژی برتر از فایل 15m
    {
        "symbol": "ETHUSDT",
        "name": "ETH_15m_1",
        "tf": "15m",
        "macd_fast": 8,
        "macd_slow": 16,
        "macd_signal": 4,
        "rsi_len": 14,
        "rsi_bull": 40,
        "rsi_bear": 70,
        "ema_trend_len": 50,
    },
    {
        "symbol": "ETHUSDT",
        "name": "ETH_15m_2",
        "tf": "15m",
        "macd_fast": 12,
        "macd_slow": 31,
        "macd_signal": 10,
        "rsi_len": 14,
        "rsi_bull": 45,
        "rsi_bear": 60,
        "ema_trend_len": 125,
    },
]

# ==========================
# حلقه اصلی
# ==========================

def main():
    # Ensure equity is numeric even if INITIAL_EQUITY ends up as a string from env loading
    try:
        equity = float(INITIAL_EQUITY)
    except (TypeError, ValueError):
        raise ValueError("BOT_INITIAL_EQUITY must be a numeric value.")
    positions = []
    last_indices = {}

    symbols = sorted(set(s["symbol"] for s in STRATEGIES))
    combo_counts = summarize_combos(STRATEGIES)
    total_combos = sum(combo_counts.values())

    print(f"[INIT] Starting equity from .env BOT_INITIAL_EQUITY={equity:.2f} USDT")
    print(f"[INIT] Tracking symbols: {', '.join(symbols)}")
    print(
        f"[INIT] Strategy combos: {total_combos} setups across "
        f"{len(combo_counts)} symbol/TF pairs"
    )
    for (symbol, tf), count in sorted(combo_counts.items()):
        print(f"[INIT]  - {symbol} {tf}: {count} setup(s)")

    print(f"[INIT] Starting equity from .env BOT_INITIAL_EQUITY={equity:.2f} USDT")

    if TELEGRAM_ENABLED:
        send_telegram(
            "سلام 👋\n"
            f"ربات *MACD + RSI Multi-Asset* برای نمادها: {', '.join(symbols)} "
            f"با سرمایه `{equity:.2f} USDT` استارت شد."
        )
    else:
        print("[WARN] Telegram is not configured; only console output.")

    while True:
        # برای هر ترکیب (symbol, tf) یکبار دیتا می‌گیریم
        by_symbol_tf = {}
        for strat in STRATEGIES:
            key = (strat["symbol"], strat["tf"])
            by_symbol_tf.setdefault(key, []).append(strat)

        for (symbol, tf), strat_list in by_symbol_tf.items():
            try:
                df_raw = fetch_klines(symbol, tf, limit=500)
            except Exception as e:
                print(f"[ERROR] fetch {symbol} {tf}: {e}")
                continue

            for strat in strat_list:
                try:
                    df = add_indicators(df_raw, strat)
                    if len(df) == 0:
                        continue

                    key = (symbol, tf, strat["name"])
                    last_idx_prev = last_indices.get(key, len(df) - WARMUP_BARS - 1)

                    last_idx_new, positions, equity = generate_signals_and_trades(
                        symbol, df, strat, last_idx_prev, positions, equity
                    )
                    last_indices[key] = last_idx_new
                except Exception as e:
                    print(f"[ERROR] strat {strat['name']} {symbol} {tf}: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
