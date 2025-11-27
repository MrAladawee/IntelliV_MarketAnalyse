############## imports ##############

import os, asyncio
from telethon import TelegramClient, events
import re
from config import TELEGRAM_TOKEN, CHAT_IDS
import requests
import pandas as pd
import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import time
import torch
import joblib
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
import torch.nn as nn
from pathlib import Path
import csv
from filelock import FileLock

#########################################


############## presettings ##############

from config import API_ID, API_HASH, PHONE

api_id = API_ID
api_hash = API_HASH
phone = PHONE

TICKERS         = ['ABIO','AFKS','AFLT','AKRN','ALRS','APTK','AQUA','ASTR','BELU','BSPB', 
                  'CBOM','CHMF','DATA','DELI','DIAS','ELFV','ENPG','EUTR','FEES','FESH', 
                  'FLOT','GAZP','GEMC','GMKN','HEAD','HNFG','HYDR','IRAO','IVAT','KMAZ', 
                  'LEAS','LENT','LKOH','LSRG','MAGN','MBNK','MDMG','MOEX','MRKC','MRKP', 
                  'MRKS','MRKU','MRKV','MRKZ','MSNG','MSRS','MTLR','MTSS','MVID','NLMK', 
                  'NVTK','OGKB','OZPH','PHOR','PIKK','PLZL','POSI','PRMD','RASP','RENI', 
                  'RNFT','ROSN','RTKM','RUAL','SBER','SELG','SFIN','SGZH','SMLT','SNGS', 
                  'SOFL','SVAV','SVCB','TATN','TGKA','TGKN','TRMK','TRNFP','T','UGLD','UPRO',
                  'VKCO','VSEH','VSMO','VTBR','WUSH','X5','YDEX']

TICKERS_FULL    = ['ABIO','AFKS','AFLT','AKRN','ALRS','APTK','AQUA','ASTR','BELU','BSPB', 
                  'CBOM','CHMF','DATA','DELI','DIAS','ELFV','ENPG','EUTR','FEES','FESH', 
                  'FLOT','GAZP','GEMC','GMKN','HEAD','HNFG','HYDR','IRAO','IVAT','KMAZ', 
                  'LEAS','LENT','LKOH','LSRG','MAGN','MBNK','MDMG','MOEX','MRKC','MRKP', 
                  'MRKS','MRKU','MRKV','MRKZ','MSNG','MSRS','MTLR','MTSS','MVID','NLMK', 
                  'NVTK','OGKB','OZPH','PHOR','PIKK','PLZL','POSI','PRMD','RASP','RENI', 
                  'RNFT','ROSN','RTKM','RUAL','SBER','SELG','SFIN','SGZH','SMLT','SNGS', 
                  'SOFL','SVAV','SVCB','TATN','TGKA','TGKN','TRMK','TRNFP','T','UGLD','UPRO',
                  'VKCO','VSEH','VSMO','VTBR','WUSH','X5','YDEX', "MTLRP", "SNGSP", "TATNP", "RTKMP", "SBERP"]

SPECIAL_TICKERS = {
    "SBER": ["SBERP"],
    "RTKM": ["RTKMP"],
    "TATN": ["TATNP"],
    "SNGS": ["SNGSP"],
    "MTLR": ["MTLRP"]
}

TELEGRAM_TOKEN = TELEGRAM_TOKEN
CHAT_ID = CHAT_IDS

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

client = TelegramClient(phone, api_id=api_id, api_hash=api_hash, system_version='4.16.30-vxCUSTOM') # Такая системная версия нужна для устранения вылетов с сессии

#########################################

# ================ Model & scalers ================
SCALER_X_PATH = os.path.join("scaler_X.pkl")
SCALER_Y_PATH = os.path.join("scaler_y.pkl")
MODEL_PATH    = os.path.join("clean_news_30d_LSTM.pth")
MODEL_NAME    = "DeepPavlov/rubert-base-cased"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler_X   = joblib.load(SCALER_X_PATH)
scaler_y   = joblib.load(SCALER_Y_PATH)
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME)

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

def takeMessage(text):

    tags = re.findall(r'#\S+(?=\s|$)', text)
    
    clean_text = text
    for tag in tags:
        clean_text = clean_text.replace(tag, '')
        clean_text = clean_text.replace(r'(?!)', '')

    clean_text = re.sub(r'[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF\ufe0f\u2600-\u26FF]*\s*#\S+(?=\s|$)', '', text)
    clean_text = re.sub(r'\s{2,}', ' ', clean_text).strip()

    return [tags, clean_text]

def sendMessageToTelegram(news, ticker, prediction, link, edit = False):

    '''
    text - only news text
    tag - only tag
    prediction:
            "ticker": ticker,
            "pred_price": float(pred_price),
            "pred_log_return": float(pred_log_ret),
            "last_price": float(last_price)
    '''

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    api_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

    broker_url = f"https://www.tbank.ru/invest/stocks/{ticker}/"                 #  NEED USE

    last_price = prediction['last_price']
    pred_price = prediction['pred_price']
    pct = pred_price*100/last_price
    pct = abs(100 - pct)

    direction = "📈" if pct > 0 else "📉"                                       #  NEED USE
    stop_loss = last_price * (1 - pct * 0.5 / 100)                               #  NEED USE

    raw_T_str = str(last_price)
    if "." in raw_T_str:
        dec = len(raw_T_str.split(".", 1)[1])
    else:
        dec = 2

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    link = f'<a href="{link}">{news}</a>'

    lines = [
        f'MarketTwits',
        f'',
        f'{direction} <a href="{broker_url}"><b>{ticker}</b></a>',
        f'⏰ {date_str} {time_str}',
        "",
        f'Текущая цена: {last_price:.{dec}f} руб.',
        f'Прогноз: {pred_price:.{dec}f} руб. ({pct:+.2f}%)',
        f'Стоп-лосс: {stop_loss:.{dec}f} руб.',
        "",
        f'📰 Новость: {link}',
        "",
    ]
    msg = "\n".join(lines)

    # Путь к фото по тикеру
    photo_path = os.path.join(BASE_DIR, "photo_ticker", f"{ticker}_photo.jpg")
    #print(f"Путь к фото: {photo_path}")
    if not os.path.exists(photo_path):
        photo_path = os.path.join(BASE_DIR, "img.jpg")
    
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

def safe_get(url, **kwargs):
    """Обёртка session.get с повтором на SSLError и логом прочих ошибок."""
    try:
        return session.get(url, timeout=10, **kwargs)
    except requests.exceptions.SSLError as e:
        time.sleep(5)
        return session.get(url, timeout=10, **kwargs)
    except Exception as e:
        print(f"[WARN] HTTP error accessing {url}: {e}", flush=True)
        return None

def get_news_price(ticker):
    """
    Берём минутные свечи, начиная за полчаса до часа news_dt,
    и возвращаем цену, ближайшую ко времени news_dt.
    """

    news_dt = datetime.datetime.now().replace(microsecond=0)

    if news_dt.weekday() == 5:
        news_dt = news_dt - datetime.timedelta(days=1)
    elif news_dt.weekday() == 6:
        news_dt = news_dt - datetime.timedelta(days=2)
    
    date_str = news_dt.date().isoformat()
    hour_mark = news_dt.replace(minute=0, second=0, microsecond=0)
    start_dt  = hour_mark - datetime.timedelta(hours=6)

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
    
    best = min(
        data,
        key=lambda c: abs(news_dt.timestamp() - datetime.datetime.fromisoformat(c[6]).timestamp())
    )

    return float(best[1])

def takePrices(tag):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    prices_file = os.path.join(BASE_DIR, "data_ticker", f'{tag}_data.csv')

    df = pd.read_csv(prices_file)
    #print(df.head(2))
    curr_date = datetime.datetime.now()
    #print(curr_date.strftime('%Y-%m-%d'))
    #print((curr_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d'))

    prices = df['CLOSE'].tail(n=30).tolist()
    prices.append(get_news_price(tag))

    #print(prices)

    return prices

def makePredict(ticker, news_text):

    # Recieve all prices for prediction
    prices = takePrices(ticker)
    if len(prices) < 31:
        #notify_error()
        return None
    
    # Log-returns
    # Явное вычисление лог-доходностей T_i / T_{i-1}

    #print(prices)

    log_rets = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        log_rets.append(np.log(curr / prev))
    log_rets = np.array(log_rets, dtype=float)  # shape (30,)

    log_cols = [f'log_ret_T-{i}' for i in range(29, 0, -1)] + ['log_ret_T']
    X_df = pd.DataFrame([log_rets], columns=log_cols)

    X_n = scaler_X.transform(X_df)
    X_t = torch.tensor(X_n, dtype = torch.float32, device=DEVICE)

    # Text
    text = f"[TICKER] {ticker} [SEP] {news_text}"
    enc  = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    # Prediction
    with torch.no_grad():
        pred_norm = model(ids, mask, X_t).cpu().item()
    pred_log_ret = scaler_y.inverse_transform([[pred_norm]])[0, 0]
    
    # Retransform
    last_price = prices[-1]
    pred_price = last_price * np.exp(pred_log_ret)

    # Return
    return {
        "ticker": ticker,
        "pred_price": float(pred_price),
        "pred_log_return": float(pred_log_ret),
        "last_price": float(last_price)
    }

def savePredict(ticker, news_text, news_url, prices, pred_price):

    '''
    ticker: 'SBER'
    news_text: 'Сбербанк увеличил прибыль..'
    news_url: 't.me/fin_news/345'
    prices: список длиной 31 (T-30, T-29, ..., T-1, T)
    pred_price: float
    '''

    BASE_DIR = Path(__file__).parent
    out_path = BASE_DIR / "prod_data_MT" / f"{ticker}_data.csv"

    print(f"[!!] Путь файла {out_path.resolve()}")

    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    row = {
        "ticker": ticker,
        "date": date_str,
        "time": time_str,
        "news": news_text,
        "news_url": news_url,
    }

    for i in range(0, 31):
        key = f"T-{i}" if i > 0 else "T"
        row[key] = prices[-(i+1)]

    row["T+1"] = pred_price
    row["pct_change"] = (pred_price / prices[-1] - 1) * 100

    # Создаём директорию, если не существует
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_path.exists()
    lock = FileLock(str(out_path) + ".lock")
    with lock:
        with out_path.open(mode='a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not out_path.exists():
                writer.writeheader()
            writer.writerow(row)
    print(f"✅ Данные для {ticker} сохранены")


#########################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#########################################

TELEGRAM_CHANNEL = 'markettwits'

# Обработчик для новых сообщений
@client.on(events.NewMessage(chats=[f'@{TELEGRAM_CHANNEL}']))
async def handler(event):

    text = event.message.message
    if not text:
        return
    
    print("\n")
    print("-------------------------Новое сообщение!")
    
    tags, clean_text = takeMessage(text)

    for tag in tags:
        #print(f"{tags} - все теги")

        if "#" in tag:
            
            #print(f"\"{tag}\" - проверяемый хештег")
            check_tag = tag[str(tag).index("#") + 1:]

            if check_tag in TICKERS_FULL:

                print("TRUE MESSAGE")
                print(f"Тикер найден: \"{check_tag}\"")
                print(f"https://t.me/@{TELEGRAM_CHANNEL}/{event.message.id}")

                try:
                    prices = takePrices(check_tag)
                    if not prices or len(prices) < 31:
                        print(f"[!] Ошибка: некорректные цены для {check_tag}: {prices}")
                        return

                    pred = makePredict(check_tag, clean_text)
                    if pred is None or "pred_price" not in pred:
                        print(f"[!] Ошибка: makePredict не вернул pred_price")
                        return

                    savePredict(
                        check_tag,
                        clean_text,
                        f"https://t.me/{TELEGRAM_CHANNEL}/{event.message.id}",
                        prices,
                        pred["pred_price"]
                    )

                except Exception as e:
                    print(f"[!] Ошибка при обработке {check_tag}: {e}")
                
                last_price = pred['last_price']
                pred_price = pred['pred_price']
                pct = pred_price*100/last_price
                pct = abs(100 - pct)

                if pct>0.5: sendMessageToTelegram(clean_text, check_tag, pred, f"https://t.me/@{TELEGRAM_CHANNEL}/{event.message.id}")

# Новый обработчик для редактированных сообщений
@client.on(events.MessageEdited(chats=[f'@{TELEGRAM_CHANNEL}']))
async def handler_edit(event):
    text = event.message.message
    if not text:
        return
    
    print("\n")
    print("-------------------------Сообщение было отредактировано!")
    
    tags, clean_text = takeMessage(text)

    print(f"{tags} - Теги после редактирования")
    print(f"{clean_text} - Текст после редактирования ({type(clean_text)})")
    
    for tag in tags:
        #print(f"{tags} - все теги")

        if "#" in tag:
            
            #print(f"\"{tag}\" - проверяемый хештег")
            check_tag = tag[str(tag).index("#") + 1:]

            if check_tag in TICKERS_FULL:

                print("TRUE MESSAGE")
                print(f"Тикер найден: \"{check_tag}\"")
                
                #sendMessageToTelegram(clean_text, check_tag, edit = True)


#########################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#########################################

def takePrices2(tag):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    prices_file = os.path.join(BASE_DIR, "data_ticker", f'{tag}_data.csv')

    df = pd.read_csv(prices_file)
    #print(df.head(2))
    curr_date = datetime.datetime.now()
    #print(curr_date.strftime('%Y-%m-%d'))
    #print((curr_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d'))

    prices = df['CLOSE'].tail(n=30).tolist()
    prices.append(get_news_price(tag))

    print(prices)

    return prices

def makePredict2(ticker, news_text):

    # Recieve all prices for prediction
    prices = takePrices2(ticker)
    if len(prices) < 31:
        #notify_error()
        return None
    
    # Log-returns
    # Явное вычисление лог-доходностей T_i / T_{i-1}

    #print(prices)

    log_rets = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        log_rets.append(np.log(curr / prev))
    log_rets = np.array(log_rets, dtype=float)  # shape (30,)

    log_cols = [f'log_ret_T-{i}' for i in range(29, 0, -1)] + ['log_ret_T']
    X_df = pd.DataFrame([log_rets], columns=log_cols)

    X_n = scaler_X.transform(X_df)
    X_t = torch.tensor(X_n, dtype = torch.float32, device=DEVICE)

    # Text
    text = f"[TICKER] {ticker} [SEP] {news_text}"
    enc  = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    ids  = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)

    # Prediction
    with torch.no_grad():
        pred_norm = model(ids, mask, X_t).cpu().item()
    pred_log_ret = scaler_y.inverse_transform([[pred_norm]])[0, 0]
    
    # Retransform
    last_price = prices[-1]
    pred_price = last_price * np.exp(pred_log_ret)

    # Return
    return {
        "ticker": ticker,
        "pred_price": float(pred_price),
        "pred_log_return": float(pred_log_ret),
        "last_price": float(last_price)
    }

async def main():
    await client.start()
    print("Юзербот запущен...")
    #print(takePrices("SBER"))
    #print(get_news_price("SBER"))
    #predict = makePredict2("SOFL", "Субхолдинг \"Софтлайна\" FabricaONE.АI выбрал организаторов планируемого IPO — Интерфакс")
    #print(predict)
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())