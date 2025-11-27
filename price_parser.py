import os
import requests
import datetime
from datetime import timedelta
import pandas as pd

# Configuration
DATA_DIR = "data_ticker"
API_BASE = "https://iss.moex.com/iss/history/engines/stock/markets/shares/securities/{ticker}.json"
CHUNK_DAYS = 30
FALLBACK_START = datetime.date(2001, 1, 1)


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def list_tickers_from_files():
    """
    Поиск всех тикеров по шаблону TICKER_data.csv в папке DATA_DIR.
    """
    tickers = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith("_data.csv"):
            ticker = filename.replace("_data.csv", "")
            tickers.append(ticker)
    return sorted(set(tickers))


def fetch_data_range(ticker, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    all_rows = []
    current = start_date
    while current <= end_date:
        chunk_end = min(end_date, current + timedelta(days=CHUNK_DAYS))
        params = {
            "from": current.strftime("%Y-%m-%d"),
            "till": chunk_end.strftime("%Y-%m-%d"),
            "iss.meta": "off",
            "iss.only": "history",
            "history.columns": "TRADEDATE,CLOSE",
        }
        url = API_BASE.format(ticker=ticker)
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get('history', {}).get('data', [])
        for rec in data:
            all_rows.append(rec)
        current = chunk_end + timedelta(days=1)

    if not all_rows:
        return pd.DataFrame(columns=["TRADEDATE", "CLOSE"])

    df = pd.DataFrame(all_rows, columns=["TRADEDATE", "CLOSE"])
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"]).dt.date
    df["CLOSE"] = pd.to_numeric(df["CLOSE"], errors="coerce")
    df = df.dropna(subset=["CLOSE"])
    df = df.drop_duplicates(subset=["TRADEDATE"], keep="last").sort_values("TRADEDATE")
    return df


def update_ticker(ticker: str) -> int:
    filepath = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    if os.path.exists(filepath):
        old = pd.read_csv(filepath, parse_dates=["TRADEDATE"])
        old["TRADEDATE"] = old["TRADEDATE"].dt.date
        start_date = old["TRADEDATE"].max() + timedelta(days=1)
    else:
        old = pd.DataFrame(columns=["TRADEDATE", "CLOSE"])
        start_date = FALLBACK_START

    end_date = datetime.date.today() - timedelta(days=1)
    if start_date > end_date:
        print(f"{ticker}: нет новых данных (последняя дата {start_date - timedelta(days=1)})")
        return 0

    df_new = fetch_data_range(ticker, start_date, end_date)
    if df_new.empty:
        print(f"{ticker}: новые данные не найдены.")
        return 0

    df_combined = pd.concat([old, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["TRADEDATE"], keep="last").sort_values("TRADEDATE")
    df_combined.to_csv(filepath, index=False)

    print(f"{ticker}: обновлён ({len(df_new)} строк добавлено)")
    return len(df_new)


def main():
    ensure_data_dir()
    tickers = list_tickers_from_files()
    print(f"Найдено тикеров для обновления: {len(tickers)}")

    total_updated = 0
    total_skipped = 0
    for ticker in tickers:
        try:
            added = update_ticker(ticker)
            if added > 0:
                total_updated += 1
            else:
                total_skipped += 1
        except Exception as e:
            print(f"{ticker}: ошибка при обновлении → {e}")
            total_skipped += 1

    print(f"\nИтог: обновлено {total_updated} тикеров, пропущено/без изменений — {total_skipped}")


if __name__ == "__main__":
    main()
