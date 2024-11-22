# Импорт библиотек
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web

# Задаем параметры
START_DATE = '2014-01-01'
END_DATE = '2018-12-31'
N_DAYS = 90  # Продолжительность периода в днях

# 1. Загрузка данных для 13-недельного казначейского векселя (^IRX) с Yahoo Finance
df_rf = yf.download('^IRX', start=START_DATE, end=END_DATE)

# Перевод данных в ежемесячную периодичность, взяв последнее значение каждого месяца
rf_13week = df_rf.resample('ME').last().Close / 100

# Расчет безрисковой доходности (ежемесячная)
rf_13week = (1 / (1 - rf_13week * N_DAYS / 360)) ** (1 / N_DAYS)
rf_13week = (rf_13week ** 30) - 1

# 2. Загрузка данных из FRED для трехмесячного казначейского векселя (TB3MS)
rf_3month = web.DataReader('TB3MS', 'fred', start=START_DATE, end=END_DATE)

# Преобразование безрисковой ставки в ежемесячные значения
rf_3month = (1 + (rf_3month / 100)) ** (1 / 12) - 1

# Построение графиков безрисковых ставок
plt.figure(figsize=(12, 6))
plt.plot(rf_13week, label='Безрисковая ставка (13-недельный казначейский вексель)', color='blue')
plt.plot(rf_3month, label='Безрисковая ставка (3-месячный казначейский вексель)', color='orange')
plt.title('Сравнение безрисковых ставок')
plt.xlabel('Дата')
plt.ylabel('Безрисковая ставка')
plt.legend()
plt.grid()
plt.show()