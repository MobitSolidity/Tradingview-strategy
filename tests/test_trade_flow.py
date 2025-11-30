from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import ETH_BTC_MTF_LIVE as bot
from strategy_config import StrategyConfig


def setup_module(module):
    bot.reset_runtime_state()


def test_generate_and_close_trade(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "DATA_DIR", tmp_path)
    monkeypatch.setattr(bot, "TRADES_LOG_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(bot, "EQUITY_LOG_PATH", tmp_path / "equity.jsonl")
    bot.reset_runtime_state()
    bot.TELEGRAM_ENABLED = False
    bot.RISK_RULES.clear()
    bot.RISK_RULES[("BTCUSDT", "15m")] = bot.RiskRules(
        min_atr=0.1, min_stop_distance=0.1, max_notional=20000, max_qty=20000
    )

    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "open_time": [base_time, base_time + timedelta(minutes=15), base_time + timedelta(minutes=30)],
            "close": [100.0, 105.0, 104.0],
            "high": [101.0, 106.0, 105.0],
            "low": [99.0, 104.0, 103.0],
            "macd": [-0.6, 0.2, 0.1],
            "signal": [-0.5, 0.0, 0.05],
            "rsi": [25.0, 35.0, 40.0],
            "emaT": [99.0, 100.0, 102.0],
            "atr": [1.0, 1.0, 1.0],
        }
    )

    strat = StrategyConfig(
        symbol="BTCUSDT",
        name="test",
        tf="15m",
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        rsi_len=14,
        rsi_bull=30,
        rsi_bear=70,
        ema_trend_len=21,
    )

    last_idx, positions, equity = bot.generate_signals_and_trades(
        "BTCUSDT", df, strat, last_idx=0, positions=[], equity=1000.0
    )

    assert last_idx == len(df) - 1
    assert len(positions) == 1
    position = positions[0]
    assert position.direction == "LONG"
    assert position.status == "OPEN"
    assert pytest.approx(position.qty, rel=1e-5) == 10.0

    exit_df = pd.DataFrame(
        {
            "open_time": [base_time + timedelta(minutes=45)],
            "high": [108.0],
            "low": [105.0],
        }
    )

    updated_equity = bot.process_exits("BTCUSDT", "15m", exit_df, positions, equity)
    assert updated_equity == pytest.approx(1020.0)
    assert position.status == "CLOSED"
    assert position.close_price == pytest.approx(position.tp)
