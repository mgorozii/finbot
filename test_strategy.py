import numpy as np

from strategy import calculate_profit


def test_simple_growth():
    # сегодня $100, завтра $200
    # купить на $1000 → 10 акций, продать → $2000
    # прибыль: $1000
    current_price = 100.0
    forecast = np.array([200.0])
    amount = 1000.0

    strategy, profit = calculate_profit(forecast, amount, current_price)

    assert profit == 1000.0
    assert "купить по $100.00" in strategy
    assert "продать по $200.00" in strategy


def test_no_profit():
    # цена падает - не покупаем
    current_price = 200.0
    forecast = np.array([100.0])
    amount = 1000.0

    strategy, profit = calculate_profit(forecast, amount, current_price)

    assert profit == 0.0
    assert "нет выгодных возможностей" in strategy


def test_two_trades():
    # две сделки: $100→$150, потом $100→$150
    # 1) купить $100 (10 акций) → продать $150 = $1500 (прибыль $500)
    # 2) купить $100 (15 акций) → продать $150 = $2250 (прибыль $750)
    # итого: $1250
    current_price = 100.0
    forecast = np.array([150.0, 100.0, 150.0])
    amount = 1000.0

    strategy, profit = calculate_profit(forecast, amount, current_price)

    assert profit == 1250.0
    assert strategy.count("купить") == 2
    assert strategy.count("продать") == 2
