import logging

import numpy as np
import pandas as pd

from models import lstm, prophet_model, random_forest

logger = logging.getLogger(__name__)


def train_all_models(data: pd.DataFrame, forecast_days: int = 30) -> dict:
    results = {}

    logger.info("начинается обучение Random Forest...")
    rf_forecast, rf_rmse, rf_mape = random_forest.train_and_predict(data, forecast_days)
    results["Random Forest"] = {"forecast": rf_forecast, "rmse": rf_rmse, "mape": rf_mape}
    logger.info(f"Random Forest обучена: RMSE={rf_rmse:.2f}, MAPE={rf_mape:.4f}")

    logger.info("начинается обучение Prophet...")
    prophet_forecast, prophet_rmse, prophet_mape = prophet_model.train_and_predict(
        data, forecast_days
    )
    results["Prophet"] = {"forecast": prophet_forecast, "rmse": prophet_rmse, "mape": prophet_mape}
    logger.info(f"Prophet обучена: RMSE={prophet_rmse:.2f}, MAPE={prophet_mape:.4f}")

    logger.info("начинается обучение LSTM...")
    lstm_forecast, lstm_rmse, lstm_mape = lstm.train_and_predict(data, forecast_days)
    results["LSTM"] = {"forecast": lstm_forecast, "rmse": lstm_rmse, "mape": lstm_mape}
    logger.info(f"LSTM обучена: RMSE={lstm_rmse:.2f}, MAPE={lstm_mape:.4f}")

    return results


def select_best_model(results: dict) -> tuple[str, np.ndarray, float]:
    best_name = min(results.keys(), key=lambda k: results[k]["rmse"])
    best_result = results[best_name]

    all_metrics = ", ".join([f"{name} (RMSE={r['rmse']:.2f})" for name, r in results.items()])
    logger.info(
        f"выбрана модель {best_name} с наименьшим RMSE={best_result['rmse']:.2f}. сравнение: {all_metrics}"
    )

    return best_name, best_result["forecast"], best_result["rmse"]


if __name__ == "__main__":
    from data import get_stock_data

    data = get_stock_data("AAPL")
    results = train_all_models(data)

    for name, result in results.items():
        print(f"{name}: RMSE={result['rmse']:.2f}, MAPE={result['mape']:.4f}")

    best_name, best_forecast, best_rmse = select_best_model(results)
    print(f"\nлучшая модель: {best_name} (RMSE={best_rmse:.2f})")
