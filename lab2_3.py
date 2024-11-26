# Импортируем библиотеки
import pandas as pd
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# Загружаем данные
df = yf.download('GC=F',
                 start='2000-01-01',
                 end='2011-12-31',
                 interval='1d',
                 progress=False)

# Подготавливаем данные
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.resample('M').last()
df.rename(columns={'Adj Close': 'price'}, inplace=True)

# Определяем функцию для теста ADF
def adf_test(x):
    indices = ['Test Statistic', 'p-value', '# of Lags Used', '# of Observations Used']
    adf_test = adfuller(x.dropna(), autolag='AIC')
    results = pd.Series(adf_test[0:4], index=indices)
    for key, value in adf_test[4].items():
        results[f'Critical Value ({key})'] = value
    return results

# Выполняем тест ADF
adf_results = adf_test(df['price'])
print("ADF Test:")
print(adf_results)

# Определяем функцию для теста KPSS
def kpss_test(x, h0_type='c'):
    indices = ['Test Statistic', 'p-value', '# of Lags']
    kpss_test_result = kpss(x.dropna(), regression=h0_type)
    results = pd.Series(kpss_test_result[0:3], index=indices)
    for key, value in kpss_test_result[3].items():
        results[f'Critical Value ({key})'] = value
    return results

# Выполняем тест KPSS
kpss_results = kpss_test(df['price'])
print("\nKPSS Test:")
print(kpss_results)

# Создаем графики ACF и PACF
import matplotlib.pyplot as plt
N_LAGS = 40
SIGNIFICANCE_LEVEL = 0.05
fig, ax = plt.subplots(2, 1)
plot_acf(df['price'].dropna(), ax=ax[0], lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL)
plot_pacf(df['price'].dropna(), ax=ax[1], lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL)
plt.show()
