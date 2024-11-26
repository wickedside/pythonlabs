# Импорт необходимых библиотек
import pandas as pd
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Загрузка данных цен на золото с Yahoo Finance
# Используем тикер 'GC=F' для золота (Gold Futures)
df = yf.download('GC=F', start='2000-01-01', end='2011-12-31', interval='1mo')

# Переименовываем столбец 'Adj Close' в 'price'
df.rename(columns={'Adj Close': 'price'}, inplace=True)

# Удаляем ненужные столбцы и пропуски
df = df[['price']].dropna()

# Добавление скользящего среднего и стандартного отклонения
WINDOW_SIZE = 12
df['rolling_mean'] = df['price'].rolling(window=WINDOW_SIZE).mean()
df['rolling_std'] = df['price'].rolling(window=WINDOW_SIZE).std()

# Построение графика цен на золото
df[['price', 'rolling_mean', 'rolling_std']].plot(title='Цена на золото')
plt.show()

# Сезонная декомпозиция с использованием мультипликативной модели
decomposition_results = seasonal_decompose(df['price'], model='multiplicative', period=12)
decomposition_results.plot().suptitle('Мультипликативная декомпозиция', fontsize=18)
plt.show()
