import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# Определяем константы
RISKY_ASSET = 'AMZN'
MARKET_BENCHMARK = '^GSPC'
START_DATE = '2014-01-01'
END_DATE = '2018-12-31'
N_DAYS = 90

# Загружаем исторические данные для рискованного актива и рыночного бенчмарка
df = yf.download([RISKY_ASSET, MARKET_BENCHMARK],
                  start=START_DATE,
                  end=END_DATE,
                  progress=False)

# Подготовка данных для анализа CAPM
X = df['Adj Close'].rename(columns={RISKY_ASSET: 'asset',
                                     MARKET_BENCHMARK: 'market'})\
                   .resample('M') \
                   .last() \
                   .pct_change() \
                   .dropna()

# Расчет ковариации и беты
covariance = X.cov().iloc[0, 1]
benchmark_variance = X.market.var()
beta = covariance / benchmark_variance

# Подгонка модели CAPM
y = X.pop('asset')
X = sm.add_constant(X)
capm_model = sm.OLS(y, X).fit()
print(capm_model.summary())

# Загружаем данные безрисковой ставки
df_rf = yf.download('^IRX', start=START_DATE, end=END_DATE)
rf = df_rf['Close'].resample('M').last() / 100
rf = (1 / (1 - rf * N_DAYS / 360))**(1 / N_DAYS)
rf = (rf ** 30) - 1

# --- Исправление: Визуализация ---
plt.figure(figsize=(10, 5))
plt.plot(rf, label='Risk-free rate (13 Week Treasury Bill)', color='blue')
plt.title('Risk-free rate (13 Week Treasury Bill)')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.legend()
plt.grid(True)
plt.show()

# Альтернативная безрисковая ставка из FRED
rf_fred = web.DataReader('TB3MS', 'fred', start=START_DATE, end=END_DATE)
rf_fred = (1 + (rf_fred / 100)) ** (1 / 12) - 1

# Построение графика для 3-месячной казначейской облигации
plt.figure(figsize=(10, 5))
plt.plot(rf_fred, label='Risk-free rate (3-Month Treasury Bill)', color='green')
plt.title('Risk-free rate (3-Month Treasury Bill)')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.legend()
plt.grid(True)
plt.show()
