import pandas as pd
import mplfinance as mpf

# чтение из csv
df = pd.read_csv('AAPL_data.csv', parse_dates=['Date'], index_col='Date')

# сериализация
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# 20-дневная SMA
df['SMA20'] = df['Close'].rolling(window=20).mean()

# 20-дневная EMA
df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()

# доп графики для sma и ema
apd = [
    mpf.make_addplot(df['SMA20'], color='red', title='SMA 20'),
    mpf.make_addplot(df['EMA20'], color='green', title='EMA 20')
]

# график
mpf.plot(df, type='candle', volume=True, title="Stock Price", addplot=apd, style='charles')