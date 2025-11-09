import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data, seq_length=14):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


def train_and_predict(
    data: pd.DataFrame, forecast_days: int = 30
) -> tuple[np.ndarray, float, float]:
    device = get_device()

    prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = create_sequences(scaled)
    train_size = int(len(X) * 0.8)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.cpu().numpy())

    rmse = root_mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled)

    forecast = []
    last_seq = scaled[-14:].reshape(1, 14, 1)
    last_seq = torch.FloatTensor(last_seq).to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(forecast_days):
            pred = model(last_seq)
            forecast.append(pred.cpu().item())
            pred_reshaped = pred.reshape(1, 1, 1)
            last_seq = torch.cat([last_seq[:, 1:, :], pred_reshaped], dim=1)

    forecast_array = np.array(forecast).reshape(-1, 1)
    forecast_rescaled = scaler.inverse_transform(forecast_array)

    return forecast_rescaled.flatten(), rmse, mape


if __name__ == "__main__":
    from data import get_stock_data

    data = get_stock_data("AAPL")
    forecast, rmse, mape = train_and_predict(data)

    print(f"LSTM RMSE: {rmse:.2f}")
    print(f"LSTM MAPE: {mape:.4f}")
    print(f"прогноз на 30 дней: {forecast[:5]}...")
    print(f"device: {get_device()}")
