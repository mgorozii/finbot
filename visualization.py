from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast(data: pd.DataFrame, forecast: np.ndarray, ticker: str) -> str:
    plt.figure(figsize=(12, 6))

    history_days = 60
    history = data["Close"].iloc[-history_days:]

    plt.plot(history.index, history.values, label="история", color="blue", linewidth=2)

    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i + 1) for i in range(len(forecast))]

    plt.plot(future_dates, forecast, label="прогноз", color="red", linewidth=2, linestyle="--")

    plt.title(f"{ticker} — прогноз на 30 дней", fontsize=14, weight="bold")
    plt.xlabel("дата")
    plt.ylabel("цена ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename, dpi=150)
    plt.close()

    return filename


if __name__ == "__main__":
    from data import get_stock_data
    from trainer import select_best_model, train_all_models

    data = get_stock_data("AAPL")
    results = train_all_models(data)
    _, forecast, _ = select_best_model(results)

    filename = plot_forecast(data, forecast, "AAPL")
    print(f"график сохранен: {filename}")
