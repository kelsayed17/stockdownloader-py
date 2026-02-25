"""Shared fixtures for intraday strategy tests."""

from decimal import Decimal

from stockdownloader.core.models.price import IntradayPriceData
from stockdownloader.core.models.trade import Direction
from stockdownloader.strategies.intraday.session import SessionState
from stockdownloader.indicators.volume import ExtendedSessionVWAP

_ZERO = Decimal("0")


def make_bar(
    date: str = "2025-01-15 10:00:00-05:00",
    open_: Decimal = Decimal("500"),
    high: Decimal = Decimal("502"),
    low: Decimal = Decimal("498"),
    close: Decimal = Decimal("501"),
    volume: int = 100_000,
) -> IntradayPriceData:
    """Create an IntradayPriceData bar for testing."""
    return IntradayPriceData(
        date=date,
        open=open_,
        high=high,
        low=low,
        close=close,
        adj_close=close,
        volume=volume,
    )


def make_tournament_bar(
    dt: str,
    o: float,
    h: float,
    l: float,
    c: float,
    v: int = 1000,
) -> IntradayPriceData:
    """Helper to create a 5-min bar from floats (tournament strategy tests)."""
    return IntradayPriceData(
        date=dt,
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(l)),
        close=Decimal(str(c)),
        adj_close=Decimal(str(c)),
        volume=v,
    )


def make_warmup_bars(n: int = 1200) -> list[IntradayPriceData]:
    """Generate n synthetic bars for warmup (78 bars/day)."""
    bars: list[IntradayPriceData] = []
    price = 450.0
    day_num = 0
    for i in range(n):
        bar_of_day = i % 78
        if bar_of_day == 0 and i > 0:
            day_num += 1
        date = f"2024-01-{2 + day_num:02d}"
        hour = 9 + (bar_of_day * 5 + 30) // 60
        minute = (bar_of_day * 5 + 30) % 60
        dt = f"{date} {hour:02d}:{minute:02d}:00-05:00"
        # Gentle uptrend with noise
        delta = 0.02 * (1 if i % 3 != 0 else -1)
        price += delta
        bars.append(make_tournament_bar(dt, price - 0.05, price + 0.10, price - 0.10, price, 5000))
    return bars


def make_vwap_bands(
    vwap: Decimal = Decimal("500"),
    std_dev: Decimal = Decimal("2"),
) -> ExtendedSessionVWAP:
    """Create an ExtendedSessionVWAP with symmetric bands."""
    return ExtendedSessionVWAP(
        vwap=vwap,
        std_dev=std_dev,
        upper_05=vwap + std_dev * Decimal("0.5"),
        lower_05=vwap - std_dev * Decimal("0.5"),
        upper_1=vwap + std_dev,
        lower_1=vwap - std_dev,
        upper_15=vwap + std_dev * Decimal("1.5"),
        lower_15=vwap - std_dev * Decimal("1.5"),
        upper_2=vwap + std_dev * Decimal("2"),
        lower_2=vwap - std_dev * Decimal("2"),
        upper_3=vwap + std_dev * Decimal("3"),
        lower_3=vwap - std_dev * Decimal("3"),
    )


def make_long_position_state(
    entry_price: Decimal = Decimal("500"),
    stop_loss: Decimal = Decimal("498"),
    take_profit: Decimal = Decimal("504"),
    risk_amount: Decimal = Decimal("2"),
    entry_mode: str = "PB",
) -> SessionState:
    """Create a SessionState with a long position already entered."""
    state = SessionState()
    state.reset("2025-01-15")
    state.in_position = True
    state.position_direction = Direction.LONG
    state.entry_price = entry_price
    state.stop_loss = stop_loss
    state.take_profit = take_profit
    state.pending_tp = take_profit
    state.entry_mode = entry_mode
    state.entry_bar = 12
    state.orig_sl = stop_loss
    state.risk_amount = risk_amount
    state.orb_extreme = entry_price
    return state


def make_short_position_state(
    entry_price: Decimal = Decimal("500"),
    stop_loss: Decimal = Decimal("502"),
    take_profit: Decimal = Decimal("496"),
    risk_amount: Decimal = Decimal("2"),
    entry_mode: str = "PB",
) -> SessionState:
    """Create a SessionState with a short position already entered."""
    state = SessionState()
    state.reset("2025-01-15")
    state.in_position = True
    state.position_direction = Direction.SHORT
    state.entry_price = entry_price
    state.stop_loss = stop_loss
    state.take_profit = take_profit
    state.pending_tp = take_profit
    state.entry_mode = entry_mode
    state.entry_bar = 12
    state.orig_sl = stop_loss
    state.risk_amount = risk_amount
    state.orb_extreme = entry_price
    return state
