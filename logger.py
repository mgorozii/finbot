import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def log_request(user_id: int, ticker: str, amount: float, model: str, rmse: float, profit: float):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | user={user_id} | ticker={ticker} | amount=${amount:.2f} | model={model} | rmse={rmse:.2f} | profit=${profit:.2f}\n"

    logger.debug(log_entry)

    with open("logs.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)
