import argparse
import json
import logging
import math
import os
import random
import time
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import pandas as pd
import requests  # type: ignore[import-untyped]
from dotenv import load_dotenv

from strategy_config import StrategyConfig, load_strategy_configs

# ==========================
# بارگذاری .env
# ==========================

load_dotenv(dotenv_path=".env")

LOG_LEVEL = os.getenv("BOT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
LOGGER = logging.getLogger("macd_rsi_bot")

DATA_DIR = Path(os.getenv("BOT_DATA_DIR", "data"))
TRADES_LOG_PATH = DATA_DIR / "trades.jsonl"
EQUITY_LOG_PATH = DATA_DIR / "equity.jsonl"

# ==========================
# تنظیمات کلی
# ==========================

DEFAULT_BINANCE_BASE_URL = "https://api.binance.com"
DEFAULT_USER_AGENT = "MACD-RSI-Bot/1.0 (+https://github.com/)"

WARMUP_BARS = 3          # چند کندل آخر را هر بار دوباره چک کند
POLL_INTERVAL = 60       # فاصله بین هر حلقه (ثانیه)
ATR_LEN = 14

RISK_PCT = 1.0           # درصد ریسک هر ترید از equity (مثلاً 1%)
STRATEGY_CONFIG_PATH = os.getenv("STRATEGY_CONFIG_PATH", "strategies.json")

# ==========================
# Telegram config
# ==========================

TG_ENABLED_RAW = os.getenv("TG_ENABLED")
TELEGRAM_ENABLED   = (TG_ENABLED_RAW or "true").lower() not in {"false", "0", "no"}
TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

HTTP_USER_AGENT = os.getenv("HTTP_USER_AGENT", DEFAULT_USER_AGENT)
BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", DEFAULT_BINANCE_BASE_URL)


class EnvSettings(NamedTuple):
    binance_base_url: str
    strategy_config_path: str
    http_user_agent: str
    bot_initial_equity: float
    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str

MAX_REQUEST_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0
MAX_CONSECUTIVE_FAILURES = 5
CIRCUIT_BREAK_SECONDS = 300
BACKOFF_JITTER_RATIO = 0.3
FAILURE_BACKOFF_CAP = 15.0


def _serialize_value(value):  # pragma: no cover - simple helper
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if isinstance(value, (set, tuple)):
        return list(value)
    return value


def log_event(level: str, message: str, **fields):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "message": message,
    }
    for key, value in fields.items():
        entry[key] = _serialize_value(value)

    LOGGER.log(getattr(logging, level.upper(), logging.INFO), json.dumps(entry, ensure_ascii=False))


def log_info(message: str, **fields):
    log_event("INFO", message, **fields)


def log_warn(message: str, **fields):
    log_event("WARNING", message, **fields)


def log_error(message: str, **fields):
    log_event("ERROR", message, **fields)


@dataclass
class FailureState:
    count: int = 0
    circuit_until: float = 0.0


SERVICE_FAILURES: dict[str, FailureState] = {
    "binance": FailureState(),
    "telegram": FailureState(),
}

BAR_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
NEXT_FETCH_AT: dict[tuple[str, str], float] = {}


@dataclass(frozen=True)
class RateLimitRule:
    max_requests: int
    window_seconds: float
    min_interval: float | None = None


RATE_LIMIT_RULES: dict[str, RateLimitRule] = {
    "binance:klines": RateLimitRule(max_requests=1200, window_seconds=60.0, min_interval=0.25),
    "telegram:send": RateLimitRule(max_requests=30, window_seconds=1.0, min_interval=0.1),
}

RATE_LIMIT_BUCKETS: dict[str, deque[float]] = {}


@dataclass(frozen=True)
class RiskRules:
    min_atr: float = 0.0
    min_stop_distance: float = 0.0
    max_qty: float | None = None
    max_notional: float | None = None


DEFAULT_RISK_RULES = RiskRules(
    min_atr=0.5,
    min_stop_distance=0.5,
    max_notional=2000.0,
)

RISK_RULES: dict[tuple[str, str], RiskRules] = {
    ("BTCUSDT", "15m"): RiskRules(min_atr=1.0, min_stop_distance=1.0, max_notional=1500.0),
    ("BTCUSDT", "1h"): RiskRules(min_atr=1.5, min_stop_distance=1.5, max_notional=2000.0),
    ("BTCUSDT", "4h"): RiskRules(min_atr=2.0, min_stop_distance=2.0, max_notional=2500.0),
    ("BTCUSDT", "1d"): RiskRules(min_atr=3.0, min_stop_distance=3.0, max_notional=3000.0),
    ("ETHUSDT", "15m"): RiskRules(min_atr=0.5, min_stop_distance=0.5, max_notional=1000.0),
    ("ETHUSDT", "1h"): RiskRules(min_atr=0.8, min_stop_distance=0.8, max_notional=1500.0),
    ("ETHUSDT", "4h"): RiskRules(min_atr=1.0, min_stop_distance=1.0, max_notional=2000.0),
    ("ETHUSDT", "1d"): RiskRules(min_atr=1.5, min_stop_distance=1.5, max_notional=2500.0),
}


def get_risk_rules(symbol: str, tf: str) -> RiskRules:
    return RISK_RULES.get((symbol, tf), DEFAULT_RISK_RULES)


def reset_runtime_state():
    BAR_CACHE.clear()
    NEXT_FETCH_AT.clear()
    for state in SERVICE_FAILURES.values():
        state.count = 0
        state.circuit_until = 0.0
    RATE_LIMIT_BUCKETS.clear()


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def persist_jsonl(path: Path, record: dict):
    ensure_data_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def record_trade_event(event_type: str, payload: dict):
    entry = {"event": event_type, **payload}
    persist_jsonl(TRADES_LOG_PATH, entry)


def record_equity_snapshot(equity: float, reason: str, context: dict | None = None):
    entry = {"equity": equity, "reason": reason, "timestamp": datetime.now(timezone.utc).isoformat()}
    if context:
        entry.update(context)
    persist_jsonl(EQUITY_LOG_PATH, entry)


def _validate_url(value: str, env_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SystemExit(
            f"[FATAL] {env_name} must be a valid http(s) URL; current value is not usable."
        )
    return value.rstrip("/")


def _validate_user_agent(value: str) -> str:
    if not value.strip():
        raise SystemExit("[FATAL] HTTP_USER_AGENT cannot be empty; set a descriptive identifier.")
    if len(value) > 256:
        raise SystemExit("[FATAL] HTTP_USER_AGENT is too long; keep it under 256 characters.")
    return value.strip()


def _validate_telegram_settings(
    enabled: bool, token: str, chat_id: str, enabled_raw: str | None
) -> tuple[bool, str, str]:
    token = token.strip()
    chat_id = chat_id.strip()

    if not enabled:
        return False, token, chat_id

    if not token:
        if enabled_raw is None:
            log_warn(
                "Telegram enabled by default but TG_BOT_TOKEN missing; disabling",
                guidance="Set TG_ENABLED=false to silence this warning",
            )
            return False, token, chat_id
        raise SystemExit(
            "[FATAL] TG_ENABLED is true but TG_BOT_TOKEN is missing; set the token or disable Telegram notifications."
        )
    if ":" not in token or len(token.split(":", 1)[0]) < 4:
        raise SystemExit(
            "[FATAL] TG_BOT_TOKEN format looks invalid; expected '<bot-id>:<secret>'. Do not paste secrets in logs."
        )

    if not chat_id:
        if enabled_raw is None:
            log_warn(
                "Telegram enabled by default but TG_CHAT_ID missing; disabling",
                guidance="Set TG_ENABLED=false to silence this warning",
            )
            return False, token, chat_id
        raise SystemExit(
            "[FATAL] TG_ENABLED is true but TG_CHAT_ID is missing; set a numeric chat ID or disable Telegram notifications."
        )
    if not chat_id.lstrip("-").isdigit():
        raise SystemExit("[FATAL] TG_CHAT_ID must be numeric (e.g., -1001234567890).")

    return enabled, token, chat_id


def _validate_equity(equity_env: str | None, default: float) -> float:
    if equity_env is None or equity_env.strip() == "":
        return default

    try:
        value = float(equity_env)
    except ValueError:
        raise SystemExit("[FATAL] BOT_INITIAL_EQUITY must be numeric; remove quotes or commas.")

    if value <= 0:
        raise SystemExit("[FATAL] BOT_INITIAL_EQUITY must be greater than zero.")

    return value


def validate_env_vars() -> EnvSettings:
    base_url = _validate_url(os.getenv("BINANCE_BASE_URL", DEFAULT_BINANCE_BASE_URL), "BINANCE_BASE_URL")
    config_path = os.getenv("STRATEGY_CONFIG_PATH", "strategies.json").strip()
    if not config_path:
        raise SystemExit("[FATAL] STRATEGY_CONFIG_PATH cannot be empty; point it to your strategies file.")

    telegram_enabled, telegram_token, telegram_chat_id = _validate_telegram_settings(
        TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TG_ENABLED_RAW
    )

    return EnvSettings(
        binance_base_url=base_url,
        strategy_config_path=config_path,
        http_user_agent=_validate_user_agent(HTTP_USER_AGENT),
        bot_initial_equity=_validate_equity(os.getenv("BOT_INITIAL_EQUITY"), default=1000.0),
        telegram_enabled=telegram_enabled,
        telegram_bot_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
    )


def apply_env_settings(env_settings: EnvSettings):
    global BINANCE_BASE_URL, HTTP_USER_AGENT, TELEGRAM_ENABLED, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    BINANCE_BASE_URL = env_settings.binance_base_url
    HTTP_USER_AGENT = env_settings.http_user_agent
    TELEGRAM_ENABLED = env_settings.telegram_enabled
    TELEGRAM_BOT_TOKEN = env_settings.telegram_bot_token
    TELEGRAM_CHAT_ID = env_settings.telegram_chat_id


def _should_circuit_break(service: str) -> bool:
    state = SERVICE_FAILURES.setdefault(service, FailureState())
    if state.circuit_until and time.time() < state.circuit_until:
        cooldown_left = int(state.circuit_until - time.time())
        log_warn(f"{service} circuit open; skipping request.", cooldown_left_seconds=cooldown_left)
        return True
    return False


def _pre_request_backoff(service: str):
    state = SERVICE_FAILURES.setdefault(service, FailureState())
    if state.count <= 0:
        return

    base = min(BASE_BACKOFF_SECONDS * (1 + state.count), FAILURE_BACKOFF_CAP)
    jitter = random.uniform(0, base * BACKOFF_JITTER_RATIO)
    delay = base + jitter
    log_warn(
        "Applying pre-request backoff due to consecutive failures",
        service=service,
        delay_seconds=round(delay, 2),
        consecutive_failures=state.count,
    )
    time.sleep(delay)


def _record_success(service: str):
    state = SERVICE_FAILURES.setdefault(service, FailureState())
    state.count = 0
    state.circuit_until = 0.0


def _record_failure(service: str):
    state = SERVICE_FAILURES.setdefault(service, FailureState())
    state.count += 1
    if state.count >= MAX_CONSECUTIVE_FAILURES:
        state.circuit_until = time.time() + CIRCUIT_BREAK_SECONDS
        log_warn(
            "Opening circuit breaker after consecutive failures",
            service=service,
            consecutive_failures=state.count,
            circuit_seconds=CIRCUIT_BREAK_SECONDS,
        )


def enforce_rate_limit(key: str | None):
    if not key:
        return

    rule = RATE_LIMIT_RULES.get(key)
    if not rule:
        return

    now = time.time()
    bucket = RATE_LIMIT_BUCKETS.setdefault(key, deque())

    while bucket and now - bucket[0] > rule.window_seconds:
        bucket.popleft()

    sleep_for = 0.0
    if bucket and rule.min_interval is not None:
        delta = now - bucket[-1]
        if delta < rule.min_interval:
            sleep_for = max(sleep_for, rule.min_interval - delta)

    if len(bucket) >= rule.max_requests:
        sleep_for = max(sleep_for, rule.window_seconds - (now - bucket[0]))

    if sleep_for > 0:
        log_warn(
            "Delaying request to satisfy rate limit",
            key=key,
            delay_seconds=round(sleep_for, 2),
            max_requests=rule.max_requests,
            window_seconds=rule.window_seconds,
        )
        time.sleep(sleep_for)

    bucket.append(time.time())


def _extract_retry_after_seconds(resp: requests.Response | None) -> float:
    if not resp:
        return 0.0

    retry_header = resp.headers.get("Retry-After")
    if not retry_header:
        return 0.0

    try:
        return float(retry_header)
    except ValueError:
        return 0.0


def bounded_request(
    method: str,
    url: str,
    *,
    params: dict | None = None,
    data: dict | None = None,
    timeout: int | float = 10,
    context: str = "request",
    service: str = "binance",
    rate_limit_key: str | None = None,
    headers: dict | None = None,
) -> requests.Response:
    if _should_circuit_break(service):
        raise RuntimeError(f"{service} circuit open; skipping {context}")

    _pre_request_backoff(service)
    enforce_rate_limit(rate_limit_key)

    request_headers = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        request_headers.update(headers)

    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        backoff = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
        backoff += random.uniform(0, backoff * BACKOFF_JITTER_RATIO)
        try:
            resp = requests.request(
                method,
                url,
                params=params,
                data=data,
                timeout=timeout,
                headers=request_headers,
            )
            resp.raise_for_status()
            _record_success(service)
            return resp
        except requests.HTTPError as err:
            status = err.response.status_code if err.response else "<no-status>"
            retry_after = _extract_retry_after_seconds(err.response)
            if status in {418, 429}:
                backoff = max(backoff, retry_after or BASE_BACKOFF_SECONDS * 2)
                log_warn(
                    "HTTP rate limit encountered",
                    context=context,
                    status=status,
                    backoff_seconds=round(backoff, 2),
                    attempt=attempt,
                    max_attempts=MAX_REQUEST_RETRIES,
                )
            else:
                log_warn(
                    "HTTP error during request",
                    context=context,
                    status=status,
                    attempt=attempt,
                    max_attempts=MAX_REQUEST_RETRIES,
                    error=str(err),
                )
            last_error = err
        except (requests.ConnectionError, requests.Timeout) as err:
            log_warn(
                "Network error during request",
                context=context,
                attempt=attempt,
                max_attempts=MAX_REQUEST_RETRIES,
                error=str(err),
            )
            last_error = err
        except Exception as err:  # pragma: no cover - defensive
            log_warn(
                "Unexpected error during request",
                context=context,
                attempt=attempt,
                max_attempts=MAX_REQUEST_RETRIES,
                error=str(err),
            )
            last_error = err

        if attempt < MAX_REQUEST_RETRIES:
            time.sleep(backoff)

    _record_failure(service)
    raise RuntimeError(f"{context} failed after {MAX_REQUEST_RETRIES} attempts") from last_error


def send_telegram(text: str):
    if not TELEGRAM_ENABLED:
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log_warn("Telegram disabled due to missing credentials")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        resp = bounded_request(
            "POST",
            url,
            data=payload,
            timeout=10,
            context="telegram send",
            service="telegram",
            rate_limit_key="telegram:send",
        )
        log_info("Telegram notification sent", response_status=resp.status_code)
    except Exception as e:
        log_error("Failed to send Telegram notification", error=str(e))


def notify_signal(text: str):
    log_info("Trade notification", detail=text)
    send_telegram(text)


def format_trade_open_message(
    *,
    symbol: str,
    tf: str,
    direction: str,
    strategy_name: str,
    entry: float,
    sl: float,
    tp: float,
    qty: float,
    notional: float,
    risk_value: float,
    equity: float,
    rationale: str,
) -> str:
    return (
        f"*NEW TRADE* `{symbol}` `{tf}` {direction}\n"
        f"Strategy: `{strategy_name}`\n"
        f"Entry: `{entry:.2f}` SL: `{sl:.2f}` TP: `{tp:.2f}`\n"
        f"Qty: `{qty:.4f}`  Notional: `{notional:.2f}`  Risk: `{risk_value:.2f} USDT`  Equity: `{equity:.2f} USDT`\n"
        f"{rationale}"
    )


def format_trade_close_message(pos: "Position", reason: str, equity: float) -> str:
    return (
        f"*TRADE CLOSED* `{pos.symbol}` `{pos.tf}` {reason}\n"
        f"Strategy: `{pos.strategy_name}`\n"
        f"Direction: {pos.direction}\n"
        f"Entry: `{pos.entry:.2f}`  Close: `{pos.close_price:.2f}`\n"
        f"Qty: `{pos.qty:.4f}`  PnL: `{pos.pnl:.2f} USDT`\n"
        f"New Equity: `{equity:.2f} USDT`\n"
        f"Open: {pos.open_time}  Close: {pos.close_time}"
    )


def get_initial_equity(env_settings: EnvSettings) -> float:
    equity_env = os.getenv("BOT_INITIAL_EQUITY")

    if equity_env is None or equity_env.strip() == "":
        log_warn("BOT_INITIAL_EQUITY not set; defaulting to 1000.00 USDT")

    return env_settings.bot_initial_equity


def configure_telegram(disable_telegram: bool):
    global TELEGRAM_ENABLED

    if disable_telegram:
        TELEGRAM_ENABLED = False
        log_info("Telegram disabled via CLI flag")
        return

    if TELEGRAM_ENABLED and (not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID):
        TELEGRAM_ENABLED = False
        log_warn(
            "Telegram disabled due to missing TG_BOT_TOKEN or TG_CHAT_ID",
            guidance="Set TG_ENABLED=false to silence this warning",
        )

# ==========================
# ابزارهای دیتا و اندیکاتور
# ==========================

def timeframe_to_seconds(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 60 * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 60 * 24
    return POLL_INTERVAL


def _request_klines(symbol: str, interval: str, *, limit: int = 500, start_time_ms: int | None = None) -> pd.DataFrame:
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params: dict[str, object] = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    resp = bounded_request(
        "GET",
        url,
        params=params,
        timeout=10,
        context=f"fetch klines {symbol} {interval}",
        service="binance",
        rate_limit_key="binance:klines",
    )
    data = resp.json()

    rows = []
    for row in data:
        rows.append(
            {
                "open_time": datetime.fromtimestamp(row[0] / 1000.0, tz=timezone.utc),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("open_time").reset_index(drop=True)
    return df


def _merge_cached_bars(key: tuple[str, str], existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"], keep="last")
    combined = combined.sort_values("open_time").reset_index(drop=True)
    # keep a sliding window to reduce per-iteration indicator cost
    if len(combined) > 800:
        combined = combined.tail(800).reset_index(drop=True)
        log_info("Trimmed kline cache", symbol=key[0], timeframe=key[1], size=len(combined))
    return combined


def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    key = (symbol, interval)
    cached = BAR_CACHE.get(key)

    start_time_ms: int | None = None
    fetch_limit = limit
    if cached is not None and not cached.empty:
        last_open = cached["open_time"].iloc[-1]
        start_time_ms = int(last_open.timestamp() * 1000) - 1
        fetch_limit = min(limit, 200)

    df_new = _request_klines(symbol, interval, limit=fetch_limit, start_time_ms=start_time_ms)

    if cached is None:
        BAR_CACHE[key] = df_new
    elif not df_new.empty:
        BAR_CACHE[key] = _merge_cached_bars(key, cached, df_new)
    else:
        BAR_CACHE[key] = cached

    return BAR_CACHE[key]


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


def add_indicators(df: pd.DataFrame, params: "StrategyConfig") -> pd.DataFrame:
    macd_fast = params.macd_fast
    macd_slow = params.macd_slow
    macd_signal = params.macd_signal
    rsi_len = params.rsi_len
    ema_len = params.ema_trend_len

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


def process_exits(symbol: str, tf: str, df: pd.DataFrame, positions: list["Position"], equity: float):
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

            if closed and close_price is not None and reason is not None:
                pos.status = "CLOSED"
                pos.close_time = bar_time
                pos.close_price = close_price

                if pos.direction == "LONG":
                    pos.pnl = (close_price - pos.entry) * pos.qty
                else:
                    pos.pnl = (pos.entry - close_price) * pos.qty

                equity += pos.pnl

                record_trade_event(
                    "close",
                    {
                        "symbol": pos.symbol,
                        "timeframe": pos.tf,
                        "strategy": pos.strategy_name,
                        "direction": pos.direction,
                        "entry": pos.entry,
                        "close": close_price,
                        "qty": pos.qty,
                        "pnl": pos.pnl,
                        "reason": reason,
                        "close_time": bar_time.isoformat(),
                    },
                )
                record_equity_snapshot(equity, "position_closed", {"symbol": pos.symbol, "timeframe": pos.tf})

                text = format_trade_close_message(pos, reason, equity)
                notify_signal(text)

    return equity

# ==========================
# تولید سیگنال + ورود پوزیشن
# ==========================

def generate_signals_and_trades(symbol: str,
                                df: pd.DataFrame,
                                params: "StrategyConfig",
                                last_idx: int,
                                positions: list["Position"],
                                equity: float) -> tuple[int, list["Position"], float]:
    tf = params.tf
    rsi_bull = params.rsi_bull
    rsi_bear = params.rsi_bear
    strat_name = params.name or f"{symbol}_{tf}"
    risk_rules = get_risk_rules(symbol, tf)

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

        if atr_curr < risk_rules.min_atr:
            log_warn(
                "Signal skipped: ATR below threshold",
                symbol=symbol,
                timeframe=tf,
                strategy=strat_name,
                atr=round(atr_curr, 4),
                min_atr=round(risk_rules.min_atr, 4),
            )
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
            if stop_dist < risk_rules.min_stop_distance:
                log_warn(
                    "Signal skipped: stop distance below minimum",
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strat_name,
                    stop_distance=round(stop_dist, 4),
                    min_stop_distance=round(risk_rules.min_stop_distance, 4),
                )
                continue

            qty = risk_value / stop_dist
            notional = qty * entry

            if risk_rules.max_qty is not None and qty > risk_rules.max_qty:
                log_warn(
                    "Signal skipped: qty above cap",
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strat_name,
                    qty=round(qty, 4),
                    max_qty=risk_rules.max_qty,
                )
                continue

            if risk_rules.max_notional is not None and notional > risk_rules.max_notional:
                log_warn(
                    "Signal skipped: notional above cap",
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strat_name,
                    notional=round(notional, 2),
                    max_notional=risk_rules.max_notional,
                )
                continue

            if qty > 0:
                pos = Position(symbol, tf, strat_name, "LONG", entry, sl, tp, qty, t)
                positions.append(pos)
                record_trade_event(
                    "open",
                    {
                        "symbol": symbol,
                        "timeframe": tf,
                        "strategy": strat_name,
                        "direction": "LONG",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "notional": notional,
                        "equity": equity,
                        "timestamp": t.isoformat(),
                    },
                )
                text = format_trade_open_message(
                    symbol=symbol,
                    tf=tf,
                    direction="LONG",
                    strategy_name=strat_name,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    qty=qty,
                    notional=notional,
                    risk_value=risk_value,
                    equity=equity,
                    rationale=f"(MACD up + RSI>{rsi_bull} + Trend Up)",
                )
                notify_signal(text)

        if short_signal:
            entry = c
            sl = entry + atr_curr
            tp = entry - 2.0 * atr_curr
            stop_dist = max(abs(entry - sl), 1e-6)
            if stop_dist < risk_rules.min_stop_distance:
                log_warn(
                    "Signal skipped: stop distance below minimum",
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strat_name,
                    stop_distance=round(stop_dist, 4),
                    min_stop_distance=round(risk_rules.min_stop_distance, 4),
                )
                continue

            qty = risk_value / stop_dist
            notional = qty * entry

            if risk_rules.max_qty is not None and qty > risk_rules.max_qty:
                log_warn(
                    "Signal skipped: qty above cap",
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strat_name,
                    qty=round(qty, 4),
                    max_qty=risk_rules.max_qty,
                )
                continue

            if risk_rules.max_notional is not None and notional > risk_rules.max_notional:
                log_warn(
                    "Signal skipped: notional above cap",
                    symbol=symbol,
                    timeframe=tf,
                    strategy=strat_name,
                    notional=round(notional, 2),
                    max_notional=risk_rules.max_notional,
                )
                continue

            if qty > 0:
                pos = Position(symbol, tf, strat_name, "SHORT", entry, sl, tp, qty, t)
                positions.append(pos)
                record_trade_event(
                    "open",
                    {
                        "symbol": symbol,
                        "timeframe": tf,
                        "strategy": strat_name,
                        "direction": "SHORT",
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "qty": qty,
                        "notional": notional,
                        "equity": equity,
                        "timestamp": t.isoformat(),
                    },
                )
                text = format_trade_open_message(
                    symbol=symbol,
                    tf=tf,
                    direction="SHORT",
                    strategy_name=strat_name,
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    qty=qty,
                    notional=notional,
                    risk_value=risk_value,
                    equity=equity,
                    rationale=f"(MACD down + RSI<{rsi_bear} + Trend Down)",
                )
                notify_signal(text)

    return end_idx, positions, equity

# ==========================
# Strategy configuration
# ==========================


def load_strategies_or_exit(path: str) -> list[StrategyConfig]:
    try:
        return load_strategy_configs(path)
    except Exception as exc:  # pragma: no cover - defensive startup guard
        raise SystemExit(
            f"[FATAL] Failed to load strategy configs from {path}: {exc}"
        ) from exc


def should_fetch_now(symbol: str, tf: str) -> bool:
    now = time.time()
    deadline = NEXT_FETCH_AT.get((symbol, tf), 0.0)
    if now < deadline:
        return False
    return True


def update_next_fetch(symbol: str, tf: str, df: pd.DataFrame):
    tf_seconds = timeframe_to_seconds(tf)
    if df.empty:
        NEXT_FETCH_AT[(symbol, tf)] = time.time() + POLL_INTERVAL
        return

    last_open = df["open_time"].iloc[-1]
    next_target = last_open.timestamp() + tf_seconds
    jitter = random.uniform(0, min(5.0, tf_seconds * 0.1))
    NEXT_FETCH_AT[(symbol, tf)] = max(time.time() + random.uniform(0, 2.0), next_target + jitter)


def profile_step(durations: list[tuple[str, float]], label: str, start_time: float):
    durations.append((label, time.perf_counter() - start_time))


def run_healthcheck() -> int:
    try:
        env_settings = validate_env_vars()
        apply_env_settings(env_settings)
        load_strategies_or_exit(env_settings.strategy_config_path)
    except SystemExit as exc:
        log_error("Healthcheck failed during configuration validation", error=str(exc))
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        log_error("Healthcheck failed during startup validation", error=str(exc))
        return 1

    try:
        resp = bounded_request(
            "GET",
            f"{BINANCE_BASE_URL}/api/v3/time",
            timeout=5,
            context="healthcheck server time",
            service="binance",
            rate_limit_key="binance:klines",
        )
        payload = resp.json()
        log_info("Healthcheck passed", server_time=payload.get("serverTime"))
        return 0
    except Exception as exc:
        log_error("Healthcheck request failed", error=str(exc))
        return 1


# ==========================
# حلقه اصلی
# ==========================

def main(run_once: bool = False, disable_telegram: bool = False):
    env_settings = validate_env_vars()

    apply_env_settings(env_settings)

    strategies = load_strategies_or_exit(env_settings.strategy_config_path)

    equity = get_initial_equity(env_settings)
    positions: list[Position] = []
    last_indices: dict[tuple[str, str, str], int] = {}

    symbols = sorted({s.symbol for s in strategies})

    configure_telegram(disable_telegram)
    log_info("Starting equity initialized", equity=round(equity, 2))
    record_equity_snapshot(equity, "startup", {"symbols": symbols})

    if TELEGRAM_ENABLED:
        send_telegram(
            "سلام 👋\n"
            f"ربات *MACD + RSI Multi-Asset* برای نمادها: {', '.join(symbols)} "
            f"با سرمایه `{equity:.2f} USDT` استارت شد."
        )
    else:
        log_warn("Telegram is not configured; only console output")

    while True:
        iteration_durations: list[tuple[str, float]] = []

        # برای هر ترکیب (symbol, tf) یکبار دیتا می‌گیریم
        by_symbol_tf: dict[tuple[str, str], list[StrategyConfig]] = {}
        for strat in strategies:
            key = (strat.symbol, strat.tf)
            by_symbol_tf.setdefault(key, []).append(strat)

        for (symbol, tf), strat_list in by_symbol_tf.items():
            if not should_fetch_now(symbol, tf):
                continue

            fetch_start = time.perf_counter()
            try:
                df_raw = fetch_klines(symbol, tf, limit=500)
                profile_step(iteration_durations, f"fetch {symbol} {tf}", fetch_start)
                update_next_fetch(symbol, tf, df_raw)
            except Exception as e:
                log_error("Failed to fetch klines", symbol=symbol, timeframe=tf, error=str(e))
                NEXT_FETCH_AT[(symbol, tf)] = time.time() + max(5.0, POLL_INTERVAL)
                continue

            for strat in strat_list:
                try:
                    ind_start = time.perf_counter()
                    df = add_indicators(df_raw, strat)
                    profile_step(iteration_durations, f"indicators {symbol} {tf} {strat.name}", ind_start)
                    if len(df) == 0:
                        continue

                    strat_key = (symbol, tf, strat.name)
                    last_idx_prev = last_indices.get(strat_key, len(df) - WARMUP_BARS - 1)

                    sig_start = time.perf_counter()
                    last_idx_new, positions, equity = generate_signals_and_trades(
                        symbol, df, strat, last_idx_prev, positions, equity
                    )
                    profile_step(iteration_durations, f"signals {symbol} {tf} {strat.name}", sig_start)
                    last_indices[strat_key] = last_idx_new
                except Exception as e:
                    log_error(
                        "Strategy execution failed",
                        strategy=strat.name,
                        symbol=symbol,
                        timeframe=tf,
                        error=str(e),
                    )

        if iteration_durations:
            top = sorted(iteration_durations, key=lambda x: x[1], reverse=True)[:3]
            for label, duration in top:
                if duration >= 0.5:
                    log_info("Profiling slow step", label=label, duration_seconds=round(duration, 2))

        if run_once:
            log_info("Completed single iteration; exiting due to --once")
            break

        sleep_time = POLL_INTERVAL + random.uniform(0, 2.0)
        time.sleep(sleep_time)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MACD+RSI multi-timeframe strategy loop.")
    parser.add_argument(
        "--cmd",
        choices=["run", "once", "healthcheck"],
        help="Command to execute (overrides BOT_CMD env when provided)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single iteration of data fetch & signal generation (useful for testing)",
    )
    parser.add_argument(
        "--disable-telegram",
        action="store_true",
        help="Skip Telegram notifications even if TG_BOT_TOKEN and TG_CHAT_ID are configured.",
    )
    return parser.parse_args()


def main_cli():
    args = parse_args()
    cmd = args.cmd or os.getenv("BOT_CMD", "run").lower()
    if args.once:
        cmd = "once"

    if cmd == "run":
        main(run_once=False, disable_telegram=args.disable_telegram)
    elif cmd == "once":
        main(run_once=True, disable_telegram=args.disable_telegram)
    elif cmd == "healthcheck":
        sys.exit(run_healthcheck())
    else:
        log_error("Unknown command", cmd=cmd)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
