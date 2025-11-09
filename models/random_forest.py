import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error


def create_lag_features(data: pd.DataFrame, n_lags: int = 14) -> pd.DataFrame:
    df = data[["Close"]].copy()
    for i in range(1, n_lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)
    df = df.dropna()
    return df


def train_and_predict(
    data: pd.DataFrame, forecast_days: int = 30
) -> tuple[np.ndarray, float, float]:
    df = create_lag_features(data)

    train_size = int(len(df) * 0.8)
    train, test = df[:train_size], df[train_size:]

    X_train = train.drop("Close", axis=1)
    y_train = train["Close"].values.ravel()
    X_test = test.drop("Close", axis=1)
    y_test = test["Close"].values.ravel()

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    forecast = []
    last_values = df["Close"].values[-14:].tolist()

    for _ in range(forecast_days):
        features = pd.DataFrame([last_values[-14:]], columns=X_train.columns)
        pred = float(model.predict(features)[0])
        forecast.append(pred)
        last_values.append(pred)

    return np.array(forecast), rmse, mape


if __name__ == "__main__":
    from data import get_stock_data

    data = get_stock_data("AAPL")
    forecast, rmse, mape = train_and_predict(data)

    print(f"Random Forest RMSE: {rmse:.2f}")
    print(f"Random Forest MAPE: {mape:.4f}")
    print(f"прогноз на 30 дней: {forecast[:5]}...")
