import os
import time
import json
import csv
import re
from datetime import datetime, date, timedelta

import warnings
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
import torch.nn as nn
from filelock import FileLock

# ================= Telegram config =================
from config import TELEGRAM_TOKEN, CHAT_IDS
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ================= Parser settings =================
TICKERS        = ['ABIO', 'AFKS', 'AFLT', 'AKRN', 'ALRS', 'APTK', 'AQUA', 'ASTR', 'BELU', 'BSPB', 
                  'CBOM', 'CHMF', 'DATA', 'DELI', 'DIAS', 'ELFV', 'ENPG', 'EUTR', 'FEES', 'FESH', 
                  'FLOT', 'GAZP', 'GEMC', 'GMKN', 'HEAD', 'HNFG', 'HYDR', 'IRAO', 'IVAT', 'KMAZ', 
                  'LEAS', 'LENT', 'LKOH', 'LSRG', 'MAGN', 'MBNK', 'MDMG', 'MOEX', 'MRKC', 'MRKP', 
                  'MRKS', 'MRKU', 'MRKV', 'MRKZ', 'MSNG', 'MSRS', 'MTLR', 'MTSS', 'MVID', 
                  'NLMK', 'NVTK', 'OGKB', 'OZPH', 'PHOR', 'PIKK', 'PLZL', 'POSI', 'PRMD', 'RASP', 
                  'RENI', 'RNFT', 'ROSN', 'RTKM', 'RUAL', 'SBER', 'SELG', 'SFIN', 
                  'SGZH', 'SMLT', 'SNGS', 'SOFL', 'SVAV', 'SVCB', 'TATN', 'TGKA', 
                  'TGKN', 'TRMK', 'TRNFP', 'T', 'UGLD', 'UPRO', 'VKCO', 'VSEH', 'VSMO', 'VTBR', 'WUSH', 
                  'X5', 'YDEX']

SPECIAL_TICKERS = {
    "SBER": ["SBERP"],
    "RTKM": ["RTKMP"],
    "TATN": ["TATNP"],
    "SNGS": ["SNGSP"],
    "MTLR": ["MTLRP"]
}

BASE_URL       = "https://smart-lab.ru/forum/news/{ticker}/page1/"
LAST_SEEN_FILE = "last_seen.json"
CHECK_INTERVAL = 68       # секунды
DAILY_DATA_DIR = "data_ticker"
# ================================================

# ================ HTTP session with retries ================
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

def safe_get(url, **kwargs):
    """Обёртка session.get с повтором на SSLError, логом и анти-кэш параметром."""
    # Добавляем анти-кэш параметр
    params = kwargs.get("params", {})
    params["_"] = int(time.time())  # уникальный параметр для обхода кэша
    kwargs["params"] = params

    try:
        return session.get(url, timeout=10, **kwargs)
    except requests.exceptions.SSLError:
        time.sleep(5)
        return session.get(url, timeout=10, **kwargs)
    except Exception as e:
        print(f"[WARN] HTTP error accessing {url}: {e}", flush=True)
        return None

# ================ Model & scalers ================
SCALER_X_PATH = os.path.join("..", "scaler_X.pkl")
SCALER_Y_PATH = os.path.join("..", "scaler_y.pkl")
MODEL_PATH    = os.path.join("..", "clean_news_30d_LSTM.pth")

MODEL_NAME    = "DeepPavlov/rubert-base-cased"
#MODEL_NAME = "./models/rubert"

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler_X   = joblib.load(SCALER_X_PATH)
scaler_y   = joblib.load(SCALER_Y_PATH)

tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

#tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
#bert_model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True)

class MultiModalModel(nn.Module):
    def __init__(self, bert_model, seq_len, lstm_hidden_dim=64):
        super().__init__()
        self.bert = bert_model
        self.text_dropout = nn.Dropout(0.2)
        self.norm_text   = nn.LayerNorm(bert_model.config.hidden_size)
        self.text_branch = nn.Sequential(
            nn.Linear(bert_model.config.hidden_size, 128),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=lstm_hidden_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.norm_num        = nn.LayerNorm(lstm_hidden_dim * 2)
        self.num_to_text_dim = nn.Linear(lstm_hidden_dim * 2, 128)
        self.film_gen = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 256)
        )
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask, X_num):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb  = bert_out.last_hidden_state[:, 0, :]
        cls_emb  = self.text_dropout(self.norm_text(cls_emb))
        cls_feat = self.text_branch(cls_emb)

        X_seq        = X_num.unsqueeze(-1)
        _, (h_n, _) = self.lstm(X_seq)
        num_feat     = torch.cat([h_n[0], h_n[1]], dim=1)
        num_feat     = self.norm_num(num_feat)
        num_proj     = self.num_to_text_dim(num_feat)

        film      = self.film_gen(cls_feat)
        gamma, beta = film.chunk(2, dim=1)
        num_mod   = num_proj * (1 + gamma) + beta

        fused = torch.cat([cls_feat, num_mod], dim=1)
        return self.head(fused).squeeze(-1)

model = MultiModalModel(bert_model=bert_model, seq_len=30, lstm_hidden_dim=64)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()
# ================================================

# ============= Price history utils =============
_history_cache = {}
def load_daily_history(ticker):
    if ticker in _history_cache:
        return _history_cache[ticker]
    path = os.path.join(DAILY_DATA_DIR, f"{ticker}_data.csv")
    if not os.path.exists(path):
        df = None
    else:
        df = pd.read_csv(path, parse_dates=["TRADEDATE"])
        df["TRADEDATE"] = df["TRADEDATE"].dt.date
        df = df.sort_values("TRADEDATE").drop_duplicates("TRADEDATE", keep="last")
    _history_cache[ticker] = df
    return df

def get_news_price(ticker, news_dt):
    """
    Берём минутные свечи, начиная за полчаса до часа news_dt,
    и возвращаем цену, ближайшую ко времени news_dt.
    """
    date_str = news_dt.date().isoformat()
    hour_mark = news_dt.replace(minute=0, second=0, microsecond=0)
    start_dt  = hour_mark - timedelta(minutes=30)

    resp = safe_get(
        f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}/candles.json",
        params={
            "from":     start_dt.isoformat(),
            "till":     f"{date_str}T23:59:59",
            "interval": 1,
            "iss.meta": "off"
        }
    )
    if not resp or resp.status_code != 200:
        return None
    data = resp.json().get("candles", {}).get("data", [])
    if not data:
        return None
    best = min(data, key=lambda c: abs(news_dt.timestamp() - datetime.fromisoformat(c[6]).timestamp()))
    return float(best[1])


def get_past_prices(hist, news_date):

    out = {}

    if hist is None:
        return out
    
    dates = hist["TRADEDATE"].tolist()
    idxs  = [i for i,d in enumerate(dates) if d < news_date]

    if not idxs:
        return out
    
    pos = max(idxs)

    for i in range(30):
        j = pos - i
        if j < 0:
            break
        out[f"T-{i+1}"] = float(hist.loc[j, "CLOSE"])

    return out
# ================================================

# Убедимся, что папка для выходных CSV существует
PROD_DATA_DIR = "prod_data"
os.makedirs(PROD_DATA_DIR, exist_ok=True)

# ================ CSV helpers ==================
def load_last_seen():
    return json.load(open(LAST_SEEN_FILE)) if os.path.exists(LAST_SEEN_FILE) else {}

def save_last_seen(d):
    with open(LAST_SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_rows(ticker):
    """
    Читает строки из prod_data/{ticker}_data.csv
    """
    fname = os.path.join(PROD_DATA_DIR, f"{ticker}_data.csv")
    if not os.path.exists(fname):
        return []
    with open(fname, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def save_rows(ticker, rows):
    """
    Сохраняет список dicts в prod_data/{ticker}_data.csv
    теперь с дополнительным столбцом pct_change
    """
    fname  = os.path.join(PROD_DATA_DIR, f"{ticker}_data.csv")
    fields = [
        "ticker","date","time","news","news_url",
        "T"
    ] + [f"T-{i}" for i in range(1,31)] + [
        "T+1","pct_change"
    ]

    lockfile = fname + ".lock"  # файл блокировки
    lock = FileLock(lockfile)
    with lock:
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
            
# ================================================

# =============== Prediction ====================
def predict_next_price(row):
    price_cols = [f"T-{i}" for i in range(30,0,-1)] + ["T"]
    # Приводим все к float. Если хоть один не преобразуется — выходим.
    try:
        prices = [float(row[c]) for c in price_cols]
    except (TypeError, ValueError, KeyError):
        return None

    # Явное вычисление лог-доходностей T_i / T_{i-1}
    log_rets = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        log_rets.append(np.log(curr / prev))
    log_rets = np.array(log_rets, dtype=float)  # shape (30,)

    X_n = scaler_X.transform(log_rets.reshape(1,-1))
    X_t = torch.tensor(X_n, dtype=torch.float32, device=DEVICE)

    text = f"[TICKER] {row['ticker']} [SEP] {row.get('news','')}"

    enc  = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        ln_norm = model(ids, mask, X_t).cpu().item()
    ln = scaler_y.inverse_transform([[ln_norm]])[0,0]
    T0 = float(row["T"])
    
    return T0 * float(np.exp(ln))
# ================================================

# =============== Telegram notify ==============
def notify(row):
    # Подготовка
    ticker     = row.get("ticker", "")
    broker_url = f"https://www.tbank.ru/invest/stocks/{ticker}/"
    cur_price  = float(row.get("T") or 0)
    pred       = float(row.get("T+1") or 0)
    pct        = (pred/cur_price - 1) * 100 if cur_price != 0 else 0
    date_str   = row.get("date", "")
    time_str   = row.get("time", "")
    
    # Разбираем все новости и ссылки
    titles = row["news"].split(" || ")
    urls   = row["news_url"].split(" || ")
    # Переворачиваем от новой к старой
    all_items   = list(zip(titles, urls))[::-1]
    total_count = len(all_items)

    # Показываем только пять
    items = all_items[:5]
    shown_count = len(items)

    # Определяем, сколько знаков после запятой у текущей цены
    raw_T_str = str(cur_price)
    if "." in raw_T_str:
        dec = len(raw_T_str.split(".", 1)[1])
    else:
        dec = 2

    # Формируем список новостей
    news_lines = [
        f'• <a href="{link}">{title}</a>'
        for title, link in items
    ]
    news_block = "\n".join(news_lines)

    stop_loss = cur_price * (1 - pct * 0.5 / 100)
    #relevance = "✅ Сделка релевантна" if abs(pct) > 1 else "❌ Сделка нерелевантна"

    # Составляем сообщение
    direction = "📈" if pct > 0 else "📉"
    lines = [
        f'Smart-Lab',
        f'',
        f'{direction} <a href="{broker_url}"><b>{ticker}</b></a>',
        f'⏰ {date_str} {time_str}',
        "",
        f'Текущая цена: {cur_price:.{dec}f} руб.',
        f'Прогноз: {pred:.{dec}f} руб. ({pct:+.2f}%)',
        f'Стоп-лосс: {stop_loss:.{dec}f} руб.',
        "",
        f'📰 <b>Новости (всего {total_count}, показано {shown_count})</b>:',
        news_block,
        "",
        #relevance
    ]
    msg = "\n".join(lines)

    # Отправка фото + сообщения
    api_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    
    # Путь к фото по тикеру
    photo_path = f"photo_ticker/{ticker}_photo.jpg"
    if not os.path.exists(photo_path):
        photo_path = "img.jpg"
    
    for cid in CHAT_IDS:
        try:
            # Открываем файл прямо перед каждым запросом,
            # чтобы "photo" было в начале чтения
            with open(photo_path, "rb") as photo:
                data = {
                    "chat_id":    cid,
                    "caption":    msg,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                }
                files = {"photo": photo}
                r = requests.post(api_url, data=data, files=files)
                r.raise_for_status()
        except Exception as e:
            print(f"Failed to send Telegram to {cid}: {e}")

# ================================================

# ================ Update + parse ================
def update_csv(ticker, title, url, time_str):
    today = date.today().isoformat()
    rows  = load_rows(ticker)

    def calc_pct(r):
        try:
            t0 = float(r["T"])
            t1 = float(r["T+1"])
            return round((t1/t0 - 1) * 100, 2) if t0 != 0 else None
        except:
            return None

    def maybe_notify(r):
        pct = calc_pct(r)
        if pct is not None and abs(pct) >= 0.5:
            notify(r)

    # 1) Если сегодня уже есть строка — обновляем её
    for r in rows:
        if r["ticker"] == ticker and r["date"] == today:
            # обновляем время и цены
            r["time"]       = time_str
            news_dt         = datetime.combine(date.today(),
                                    datetime.strptime(time_str, "%H:%M").time())
            r["T"]          = get_news_price(ticker, news_dt)
            # конкатенируем текст и URL новой новости
            r["news"]       += " || " + title
            r["news_url"]   += " || " + url
            # пересчитываем прогноз
            r["T+1"]        = predict_next_price(r)
            r["pct_change"] = calc_pct(r)

            save_rows(ticker, rows)
            print(f"[{datetime.now()}] {ticker}: ОБНОВЛЕНА новость «{title}», pct={r['pct_change']}%", flush=True)
            maybe_notify(r)
            return

    # 2) Иначе — создаём новую запись
    news_dt = datetime.combine(date.today(),
                datetime.strptime(time_str, "%H:%M").time())
    T0      = get_news_price(ticker, news_dt)
    past    = get_past_prices(load_daily_history(ticker), news_dt.date())

    new_row = {
        "ticker":   ticker,
        "date":     today,
        "time":     time_str,
        "news":     title,
        "news_url": url,
        "T":        T0
    }
    new_row.update(past)
    new_row["T+1"]        = predict_next_price(new_row)
    new_row["pct_change"] = calc_pct(new_row)

    rows.append(new_row)
    save_rows(ticker, rows)
    print(f"[{datetime.now()}] {ticker}: ДОБАВЛЕНА новость «{title}», pct={new_row['pct_change']}%", flush=True)
    maybe_notify(new_row)

def parse_news(ticker):
    """
    Ищем блок div.temp_block → ul.temp_headers--have-numbers → все <li>,
    фильтруем только по тем, где между </b> и <a> стоит время HH:MM.
    """
    url  = BASE_URL.format(ticker=ticker)
    resp = safe_get(url)
    if not resp or resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")

    block = soup.find("div", class_="temp_block")
    if not block:
        return []

    ul = block.find("ul", class_="temp_headers temp_headers--have-numbers")
    if not ul:
        return []

    out = []
    for li in ul.find_all("li"):
        b = li.find("b")
        if not b:
            continue
        date_str = None
        for sib in b.next_siblings:
            if getattr(sib, "name", None) == "a":
                break
            if isinstance(sib, str) and sib.strip():
                date_str = sib.strip()
                break
        if not date_str or not re.fullmatch(r"\d{1,2}:\d{2}", date_str):
            continue
        a = li.find("a")
        if not a or not a.get("href"):
            continue
        out.append({
            "title": a.get_text(strip=True),
            "url":   "https://smart-lab.ru" + a["href"],
            "time":  date_str
        })
    return out

def main():
    os.makedirs(DAILY_DATA_DIR, exist_ok=True)
    last_seen = load_last_seen()
    while True:
        for t in TICKERS:
            try:
                news = parse_news(t)
                if not news:
                    continue
                newest = news[0]["url"]
                if t not in last_seen:
                    for it in reversed(news):
                        update_csv(t, it["title"], it["url"], it["time"])
                        for sp in SPECIAL_TICKERS.get(t, []):
                            update_csv(sp, it["title"], it["url"], it["time"])
                    last_seen[t] = newest
                    save_last_seen(last_seen)
                else:
                    to_add = []
                    for it in news:
                        if it["url"] == last_seen[t]:
                            break
                        to_add.append(it)
                    if to_add:
                        for it in reversed(to_add):
                            update_csv(t, it["title"], it["url"], it["time"])
                            for sp in SPECIAL_TICKERS.get(t, []):
                                update_csv(sp, it["title"], it["url"], it["time"])
                        last_seen[t] = newest
                        save_last_seen(last_seen)

            except Exception as e:
                print(f"[ERROR] {t}: {e}", flush=True)

        print(f"[{datetime.now()}] Закончен цикл проверки для {len(TICKERS)} тикеров", flush=True)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()