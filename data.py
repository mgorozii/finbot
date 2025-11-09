import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_DAYS = 1


def get_stock_data(ticker: str) -> pd.DataFrame:
    cache_file = CACHE_DIR / f"{ticker}.csv"

    if cache_file.exists():
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age < timedelta(days=CACHE_DAYS):
            data = pd.read_csv(cache_file, index_col=0, parse_dates=[0])
            logger.info(
                f"загружены данные {ticker} из кеша: {len(data)} строк, период {data.index[0].date()} - {data.index[-1].date()}"
            )
            return data

    logger.info(f"загрузка данных {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

    if data.empty:
        raise ValueError(f"не удалось загрузить данные {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.to_csv(cache_file)
    logger.info(
        f"данные {ticker} сохранены: {len(data)} строк, период {data.index[0].date()} - {data.index[-1].date()}"
    )
    return data


if __name__ == "__main__":
    df = get_stock_data("AAPL")
    print(f"загружено {len(df)} строк")
    print(df.tail())
