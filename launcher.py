import subprocess
import time
import os
import schedule
from datetime import datetime

print("ПРОГРАММА ЗАПУЩЕНА")
print("CWD =", os.getcwd())
print("FILES =", os.listdir("."), flush=True)

def run_price_parser():
    print(f"[{datetime.now()}] Запуск price_parser.py", flush=True)
    subprocess.Popen(["python3", "price_parser.py"])

# Запуск основных сеансов
print("Запуск BotPred.py ...", flush=True)
bot = subprocess.Popen(["python3", "BotPred.py"])

print("Запуск Test.py ...", flush=True)
test = subprocess.Popen(["python3", "Test.py"])

print(f"BotPred PID: {bot.pid}", flush=True)
print(f"Test PID: {test.pid}", flush=True)

# Каждый день в 03:00
schedule.every().day.at("03:00").do(run_price_parser)

print("Планировщик запущен.", flush=True)

while True:
    schedule.run_pending()
    time.sleep(1)