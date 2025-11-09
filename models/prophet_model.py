import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error


def train_and_predict(
    data: pd.DataFrame, forecast_days: int = 30
) -> tuple[np.ndarray, float, float]:
    df = data[["Close"]].reset_index()
    df.columns = ["ds", "y"]

    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(train)

    future_test = model.make_future_dataframe(periods=len(test))
    forecast_test = model.predict(future_test)
    y_pred = forecast_test["yhat"].iloc[-len(test) :].values

    rmse = root_mean_squared_error(test["y"], y_pred)
    mape = mean_absolute_percentage_error(test["y"], y_pred)

    model_full = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model_full.fit(df)

    future = model_full.make_future_dataframe(periods=forecast_days)
    forecast = model_full.predict(future)
    predictions = forecast["yhat"].iloc[-forecast_days:].values

    return predictions, rmse, mape


if __name__ == "__main__":
    from data import get_stock_data

    data = get_stock_data("AAPL")
    forecast, rmse, mape = train_and_predict(data)

    print(f"Prophet RMSE: {rmse:.2f}")
    print(f"Prophet MAPE: {mape:.4f}")
    print(f"прогноз на 30 дней: {forecast[:5]}...")
