import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str
    name: str
    tf: str
    macd_fast: int
    macd_slow: int
    macd_signal: int
    rsi_len: int
    rsi_bull: int
    rsi_bear: int
    ema_trend_len: int


def format_strategy_key(symbol: str, tf: str, name: str) -> str:
    return f"{symbol}:{tf}:{name}"


def _validate_uniqueness(strategies: Sequence[StrategyConfig]) -> list[StrategyConfig]:
    seen: set[tuple[str, str, str]] = set()
    duplicates: list[str] = []

    for strat in strategies:
        key = (strat.symbol, strat.tf, strat.name)
        if key in seen:
            duplicates.append(format_strategy_key(*key))
            continue
        seen.add(key)

    if duplicates:
        duplicate_list = ", ".join(duplicates)
        raise ValueError(
            "Duplicate strategy definitions detected; ensure each (symbol, tf, name) "
            f"combination is unique. Offending entries: {duplicate_list}."
        )

    return list(strategies)


def _parse_strategy(raw: dict) -> StrategyConfig:
    required_fields: Iterable[str] = (
        "symbol",
        "name",
        "tf",
        "macd_fast",
        "macd_slow",
        "macd_signal",
        "rsi_len",
        "rsi_bull",
        "rsi_bear",
        "ema_trend_len",
    )

    missing = [field for field in required_fields if field not in raw]
    if missing:
        raise ValueError(f"Missing required strategy fields: {', '.join(missing)}")

    return StrategyConfig(
        symbol=str(raw["symbol"]),
        name=str(raw["name"]),
        tf=str(raw["tf"]),
        macd_fast=int(raw["macd_fast"]),
        macd_slow=int(raw["macd_slow"]),
        macd_signal=int(raw["macd_signal"]),
        rsi_len=int(raw["rsi_len"]),
        rsi_bull=int(raw["rsi_bull"]),
        rsi_bear=int(raw["rsi_bear"]),
        ema_trend_len=int(raw["ema_trend_len"]),
    )


def load_strategy_configs(path: str | Path = "strategies.json") -> list[StrategyConfig]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Strategy config file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Strategy config must be a list of strategy objects.")

    parsed: list[StrategyConfig] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Strategy entry at index {idx} must be an object/dict.")
        parsed.append(_parse_strategy(item))

    return _validate_uniqueness(parsed)
