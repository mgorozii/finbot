import numpy as np


# calculate_profit реализует жадный алгоритм: покупаем перед каждым ростом, продаем перед каждым падением
# возможно есть вариант лучше, но мне пришло в голову взять этот алгоритм с leetcode
def calculate_profit(
    forecast: np.ndarray, amount: float, current_price: float
) -> tuple[str, float]:
    full_forecast = np.concatenate([[current_price], forecast])

    if len(full_forecast) < 2:
        return "недостаточно данных для анализа", 0.0

    strategy = []
    total_profit = 0.0
    shares = 0
    buy_price = 0.0

    for i in range(len(full_forecast) - 1):
        current = full_forecast[i]
        next_price = full_forecast[i + 1]

        # если завтра цена выше и у нас нет акций - покупаем
        if next_price > current and shares == 0:
            shares = amount / current
            buy_price = current
            day_label = "сегодня" if i == 0 else f"день {i}"
            strategy.append(f"{day_label}: купить по ${current:.2f}")

        # если завтра цена ниже и у нас есть акции - продаем
        elif next_price < current and shares > 0:
            sell_price = current
            profit = shares * (sell_price - buy_price)
            total_profit += profit
            amount += profit
            shares = 0
            day_label = "сегодня" if i == 0 else f"день {i}"
            strategy.append(f"{day_label}: продать по ${sell_price:.2f}")

    # если в конце прогноза у нас есть акции - продаем по последней цене
    if shares > 0:
        last_price = full_forecast[-1]
        profit = shares * (last_price - buy_price)
        total_profit += profit
        last_day = len(full_forecast) - 1
        day_label = "сегодня" if last_day == 0 else f"день {last_day}"
        strategy.append(f"{day_label}: продать по ${last_price:.2f}")

    if not strategy:
        return "нет выгодных возможностей для торговли", 0.0

    return "\n".join(strategy), total_profit


if __name__ == "__main__":
    from data import get_stock_data
    from trainer import select_best_model, train_all_models

    data = get_stock_data("MSFT")
    results = train_all_models(data)
    _, forecast, _ = select_best_model(results)

    current_price = data["Close"].iloc[-1]
    strategy, profit = calculate_profit(forecast, 1000, current_price)

    print("стратегия:")
    print(strategy)
    print(f"\nпотенциальная прибыль: ${profit:.2f}")
