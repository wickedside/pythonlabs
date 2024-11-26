# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import (ExponentialSmoothing,
                                         SimpleExpSmoothing,
                                         Holt)

# Установка цветовой палитры
plt.set_cmap('cubehelix')
sns.set_palette('cubehelix')
COLORS = [plt.cm.cubehelix(x) for x in [0.1, 0.3, 0.5, 0.7]]

# Шаг 1: Загрузка данных акций Google
df = yf.download('GOOG',
                 start='2010-01-01',
                 end='2018-12-31',
                 progress=False)

# Шаг 2: Пересчет данных на ежемесячную периодичность
goog = df.resample('M').last().rename(columns={'Adj Close': 'adj_close'})['adj_close']

# Шаг 3: Разделение выборки на обучающую и тестовую
train_indices = goog.index.year < 2018
goog_train = goog[train_indices]
goog_test = goog[~train_indices]
test_length = len(goog_test)

# Шаг 4: График цен акций Google
goog.plot(title="Цена акций Google")
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.show()

# Шаг 5: Подгонка моделей SES и прогнозирование
# Модель 1: alpha=0.2
ses_1 = SimpleExpSmoothing(goog_train).fit(smoothing_level=0.2, optimized=False)
ses_forecast_1 = ses_1.forecast(test_length)

# Модель 2: alpha=0.5
ses_2 = SimpleExpSmoothing(goog_train).fit(smoothing_level=0.5, optimized=False)
ses_forecast_2 = ses_2.forecast(test_length)

# Модель 3: Автоматическое определение alpha
ses_3 = SimpleExpSmoothing(goog_train).fit()
alpha = ses_3.model.params['smoothing_level']
ses_forecast_3 = ses_3.forecast(test_length)

# Шаг 6: График результатов моделей SES
goog.plot(color=COLORS[0],
          title='Simple Exponential Smoothing',
          label='Фактические',
          legend=True)
ses_forecast_1.plot(color=COLORS[1], legend=True, label=r'$\alpha=0.2$')
ses_1.fittedvalues.plot(color=COLORS[1])
ses_forecast_2.plot(color=COLORS[2], legend=True, label=r'$\alpha=0.5$')
ses_2.fittedvalues.plot(color=COLORS[2])
ses_forecast_3.plot(color=COLORS[3], legend=True, label=r'$\alpha={0:.4f}$'.format(alpha))
ses_3.fittedvalues.plot(color=COLORS[3])
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.show()

# Шаг 7: Подгонка моделей Holt's Smoothing и прогнозирование
# Модель Холта с линейным трендом
hs_1 = Holt(goog_train).fit()
hs_forecast_1 = hs_1.forecast(test_length)

# Модель Холта с экспоненциальным трендом
hs_2 = Holt(goog_train, exponential=True).fit()
hs_forecast_2 = hs_2.forecast(test_length)

# Модель Холта с экспоненциальным трендом и затуханием
hs_3 = Holt(goog_train, exponential=True, damped_trend=True).fit()
hs_forecast_3 = hs_3.forecast(test_length)

# Шаг 8: График результатов моделей Holt's Smoothing
goog.plot(color=COLORS[0],
          title="Модели сглаживания Холта",
          label='Фактические',
          legend=True)
hs_1.fittedvalues.plot(color=COLORS[1])
hs_forecast_1.plot(color=COLORS[1], legend=True, label='Линейный тренд')
hs_2.fittedvalues.plot(color=COLORS[2])
hs_forecast_2.plot(color=COLORS[2], legend=True, label='Экспоненциальный тренд')
hs_3.fittedvalues.plot(color=COLORS[3])
hs_forecast_3.plot(color=COLORS[3], legend=True, label='Экспоненциальный тренд (затухающий)')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.show()

# Шаг 9: Подгонка моделей Холта-Винтерса и прогнозирование
SEASONAL_PERIODS = 12

# Модель Холта-Винтерса с экспоненциальным трендом
hw_1 = ExponentialSmoothing(goog_train,
                            trend='mul',
                            seasonal='add',
                            seasonal_periods=SEASONAL_PERIODS).fit()
hw_forecast_1 = hw_1.forecast(test_length)

# Модель Холта-Винтерса с экспоненциальным трендом и затуханием
hw_2 = ExponentialSmoothing(goog_train,
                            trend='mul',
                            seasonal='add',
                            seasonal_periods=SEASONAL_PERIODS,
                            damped_trend=True).fit()
hw_forecast_2 = hw_2.forecast(test_length)

# Шаг 10: График результатов моделей Холта-Винтерса
goog.plot(color=COLORS[0],
          title="Модели сезонного сглаживания Холта-Винтерса",
          label='Фактические',
          legend=True)
hw_1.fittedvalues.plot(color=COLORS[1])
hw_forecast_1.plot(color=COLORS[1], legend=True, label='Сезонное сглаживание')
phi = hw_2.model.params['damping_trend']
plot_label = f'Сезонное сглаживание (затухание с $\phi={phi:.4f}$)'
hw_2.fittedvalues.plot(color=COLORS[2])
hw_forecast_2.plot(color=COLORS[2], legend=True, label=plot_label)
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.show()

