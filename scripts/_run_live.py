"""Launcher: load .env credentials, then run the production runner.
Non-invasive wrapper so we don't have to patch production_runner.py.
"""
import dotenv
dotenv.load_dotenv()  # pulls ALPACA_API_KEY / ALPACA_API_SECRET / TELEGRAM_* into env
from quantcore.live import production_runner
if __name__ == "__main__":
    production_runner.main()
