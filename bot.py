import asyncio
import logging
import os
import warnings
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import BotCommand, FSInputFile, Message
from dotenv import load_dotenv

from data import get_stock_data
from logger import log_request
from strategy import calculate_profit
from trainer import select_best_model, train_all_models
from visualization import plot_forecast

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*plotly.*")

logger = logging.getLogger(__name__)

bot = Bot(token=os.getenv("BOT_TOKEN"))
dp = Dispatcher(storage=MemoryStorage())


class AnalysisState(StatesGroup):
    ticker = State()
    amount = State()


async def set_bot_commands(bot: Bot):
    commands = [
        BotCommand(command="analyze", description="начать анализ акции"),
    ]
    await bot.set_my_commands(commands)


@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("отправь тикер (например: AAPL, MSFT, TSLA)")


@dp.message(Command("analyze"))
async def cmd_analyze(message: Message, state: FSMContext):
    await message.answer("отправь тикер")
    await state.set_state(AnalysisState.ticker)


@dp.message(AnalysisState.ticker)
async def process_ticker(message: Message, state: FSMContext):
    ticker = message.text.upper().strip()
    await state.update_data(ticker=ticker)
    await message.answer("отправь сумму ($)")
    await state.set_state(AnalysisState.amount)


@dp.message(AnalysisState.amount)
async def process_amount(message: Message, state: FSMContext):
    try:
        amount = float(message.text.replace(",", "."))
        if amount <= 0:
            raise ValueError
    except ValueError:
        await message.answer("некорректная сумма, попробуй снова")
        return

    data = await state.get_data()
    ticker = data["ticker"]
    await state.clear()

    await message.answer("загружаю данные и обучаю модели, подожди...")

    try:
        stock_data = get_stock_data(ticker)
        results = train_all_models(stock_data)
        best_model, forecast, best_rmse = select_best_model(results)

        current_price = stock_data["Close"].iloc[-1]
        price_change = ((forecast[-1] - current_price) / current_price) * 100

        chart_path = plot_forecast(stock_data, forecast, ticker)
        strategy_text, profit = calculate_profit(forecast, amount, current_price)

        recommendation = ""
        if profit <= 0:
            recommendation = "\n⚠️ не рекомендуется к покупке"

        summary = (
            f"тикер: {ticker}\n"
            f"модель: {best_model}\n"
            f"RMSE: {best_rmse:.2f}\n"
            f"текущая цена: ${current_price:.2f}\n"
            f"прогноз через 30 дней: ${forecast[-1]:.2f}\n"
            f"изменение: {price_change:+.2f}%\n\n"
            f"стратегия:\n{strategy_text}\n\n"
            f"потенциальная прибыль: ${profit:.2f}{recommendation}"
        )

        photo = FSInputFile(chart_path)
        await message.answer_photo(photo, caption=summary)

        Path(chart_path).unlink()

        log_request(message.from_user.id, ticker, amount, best_model, best_rmse, profit)

    except ValueError as e:
        logger.error(f"ошибка валидации {ticker}: {e}", exc_info=True)
        await message.answer(f"ошибка: {e}")
    except Exception as e:
        logger.exception(f"ошибка {ticker} {e=}")
        await message.answer(f"произошла ошибка: {e}")


@dp.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    ticker = message.text.upper().strip()
    await state.update_data(ticker=ticker)
    await message.answer("отправь сумму ($)")
    await state.set_state(AnalysisState.amount)


async def main():
    await set_bot_commands(bot)
    logger.info("команды бота установлены")
    await dp.start_polling(bot)


if __name__ == "__main__":
    logger.info("starting...")
    asyncio.run(main())
