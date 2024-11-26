# Импорт необходимых библиотек
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as scs
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')

# Шаг 1: Загрузка данных акций Google
df = yf.download('GOOG',
                 start='2015-01-01',
                 end='2018-12-31',
                 progress=False)
goog = df.resample('W').last().rename(columns={'Adj Close': 'adj_close'})['adj_close']

# Шаг 2: Применение разности первого порядка и график
goog_diff = goog.diff().dropna()
fig, ax = plt.subplots(2, sharex=True, figsize=(12,8))
goog.plot(title="Цена акций Google", ax=ax[0])
ax[0].set_ylabel('Цена')
goog_diff.plot(ax=ax[1], title='Разности первого порядка')
ax[1].set_ylabel('Изменение цены')
plt.xlabel('Дата')
plt.show()

# Шаг 3: Проверка стационарности
def test_autocorrelation(x, lags=40, title=''):
    from statsmodels.tsa.stattools import adfuller, kpss

    # Тест Дики-Фуллера
    adf_result = adfuller(x.dropna())
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])

    # Тест KPSS
    kpss_result = kpss(x.dropna(), regression='c', nlags='auto')
    print('KPSS Statistic: %f' % kpss_result[0])
    print('p-value: %f' % kpss_result[1])

    # Построение графиков ACF и PACF
    fig, ax = plt.subplots(2,1, figsize=(12,8))
    plot_acf(x.dropna(), lags=lags, ax=ax[0])
    plot_pacf(x.dropna(), lags=lags, ax=ax[1])
    plt.tight_layout()
    plt.show()

print("Проверка стационарности разностного ряда:")
test_autocorrelation(goog_diff)

# Шаг 4: Подбор моделей ARIMA

# Модель 1: ARIMA(2,1,1)
model_1 = ARIMA(goog, order=(2, 1, 1))
arima_1 = model_1.fit()
print("Сводка модели ARIMA(2,1,1):")
print(arima_1.summary())

# Модель 2: ARIMA(3,1,2)
model_2 = ARIMA(goog, order=(3, 1, 2))
arima_2 = model_2.fit()
print("Сводка модели ARIMA(3,1,2):")
print(arima_2.summary())

# Модель 3: ARIMA(0,1,1)
model_3 = ARIMA(goog, order=(0, 1, 1))
arima_3 = model_3.fit()
print("Сводка модели ARIMA(0,1,1):")
print(arima_3.summary())

# Шаг 5: Диагностика остатков моделей
def arima_diagnostics(resids, n_lags=40, model_name=''):
    # Создание подзаголовков
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,8))
    resids = resids.dropna()
    resids_standardized = (resids - resids.mean()) / resids.std()
    # Остатки во времени
    sns.lineplot(x=np.arange(len(resids_standardized)), y=resids_standardized, ax=ax1)
    ax1.set_title(f'Стандартизированные остатки {model_name}')
    # Распределение остатков
    sns.histplot(resids_standardized, kde=True, stat="density", linewidth=0, ax=ax2)
    x_lim = (-1.96 * 2, 1.96 * 2)
    r_range = np.linspace(x_lim[0], x_lim[1], num=100)
    norm_pdf = scs.norm.pdf(r_range)
    ax2.plot(r_range, norm_pdf, 'r', lw=2, label='N(0,1)')
    ax2.set_title('Распределение стандартизированных остатков')
    ax2.legend()
    # Q-Q plot
    sm.qqplot(resids_standardized, line='s', ax=ax3)
    ax3.set_title('Q-Q график')
    # График ACF
    plot_acf(resids_standardized, ax=ax4, lags=n_lags, alpha=0.05)
    ax4.set_title('График ACF остатков')
    plt.tight_layout()
    plt.show()

arima_diagnostics(arima_1.resid, model_name='ARIMA(2,1,1)')

#arima_diagnostics(arima_2.resid, model_name='ARIMA(3,1,2)')

# Шаг 6: Тест Льюнга-Бокса для обеих моделей
# Тест Льюнга-Бокса для модели ARIMA(2,1,1)
ljung_box_results_1 = acorr_ljungbox(arima_1.resid, lags=40, return_df=True)
# Тест Льюнга-Бокса для модели ARIMA(3,1,2)
ljung_box_results_2 = acorr_ljungbox(arima_2.resid, lags=40, return_df=True)

# График результатов теста Льюнга-Бокса
fig, ax = plt.subplots(2, 1, figsize=(12,10))
sns.scatterplot(x=ljung_box_results_1.index, y=ljung_box_results_1['lb_pvalue'], ax=ax[0])
ax[0].axhline(0.05, ls='--', c='r')
ax[0].set_title("Результаты теста Льюнга-Бокса для ARIMA(2,1,1)")
ax[0].set_xlabel('Лаг')
ax[0].set_ylabel('p-value')
sns.scatterplot(x=ljung_box_results_2.index, y=ljung_box_results_2['lb_pvalue'], ax=ax[1])
ax[1].axhline(0.05, ls='--', c='r')
ax[1].set_title("Результаты теста Льюнга-Бокса для ARIMA(3,1,2)")
ax[1].set_xlabel('Лаг')
ax[1].set_ylabel('p-value')
plt.tight_layout()
plt.show()

# Шаг 7: Прогнозирование на 2019 год
# Загрузка данных за 2019 год
df_test = yf.download('GOOG',
                      start='2019-01-01',
                      end='2019-03-31',
                      progress=False)
test = df_test.resample('W').last().rename(columns={'Adj Close': 'adj_close'})['adj_close']

# Прогнозирование
n_forecasts = len(test)

# Прогноз от модели ARIMA(2,1,1)
forecast_1 = arima_1.get_forecast(steps=n_forecasts)
forecast_1_df = forecast_1.conf_int(alpha=0.05)
forecast_1_df['prediction'] = forecast_1.predicted_mean
forecast_1_df.index = test.index

# Прогноз от модели ARIMA(3,1,2)
forecast_2 = arima_2.get_forecast(steps=n_forecasts)
forecast_2_df = forecast_2.conf_int(alpha=0.05)
forecast_2_df['prediction'] = forecast_2.predicted_mean
forecast_2_df.index = test.index

# График прогнозов и фактических значений
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(test, label='Фактические значения', color='black')
ax.plot(forecast_1_df['prediction'], label='Прогноз ARIMA(2,1,1)', color='blue')
ax.fill_between(forecast_1_df.index,
                forecast_1_df.iloc[:, 0],
                forecast_1_df.iloc[:, 1], color='blue', alpha=0.1)
ax.plot(forecast_2_df['prediction'], label='Прогноз ARIMA(3,1,2)', color='green')
ax.fill_between(forecast_2_df.index,
                forecast_2_df.iloc[:, 0],
                forecast_2_df.iloc[:, 1], color='green', alpha=0.1)
ax.set_title('Прогноз цен акций Google на 2019 год')
ax.set_xlabel('Дата')
ax.set_ylabel('Цена акций')
ax.legend()
plt.show()
