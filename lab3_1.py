# Импорт библиотек
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# Настройка визуализации (используем стиль ggplot)
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = [8, 4.5]
plt.rcParams['figure.dpi'] = 300
warnings.simplefilter(action='ignore', category=FutureWarning)

# Указываем рискованный актив и временной горизонт
RISKY_ASSET = 'AMZN'
MARKET_BENCHMARK = '^GSPC'
START_DATE = '2014-01-01'
END_DATE = '2018-12-31'

# Загрузка данных с Yahoo Finance
df = yf.download([RISKY_ASSET, MARKET_BENCHMARK],
                 start=START_DATE,
                 end=END_DATE,
                 progress=False)

# Подготовка данных (ресемплирование по месяцам и расчет доходности)
X = df['Adj Close'].rename(columns={RISKY_ASSET: 'asset', MARKET_BENCHMARK: 'market'}) \
                   .resample('M') \
                   .last() \
                   .pct_change() \
                   .dropna()

# Расчет бета с использованием ковариационного подхода
covariance = X.cov().iloc[0, 1]
benchmark_variance = X['market'].var()
beta = covariance / benchmark_variance
print(f'Beta: {beta:.4f}')

# Подготовка данных для регрессии и построение модели CAPM
y = X.pop('asset')  # Доходность актива
X = sm.add_constant(X)  # Добавляем константу

# Оценка CAPM через линейную регрессию
capm_model = sm.OLS(y, X).fit()
print(capm_model.summary())