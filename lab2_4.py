import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt

# Загрузка данных о стоимости акций Google из файла CSV
df = pd.read_csv(r'./google_stock_data.csv', parse_dates=['Date'])

# Очистка названий колонок, если это необходимо
df.columns = df.columns.str.strip()  # Удаляем пробелы в названиях колонок

# Проверка наличия колонки 'adj_close' после очистки
if 'adj_close' not in df.columns:
    print("Колонка 'adj_close' не найдена! Доступные колонки:", df.columns)
else:
    # Установка индекса по колонке 'Date'
    df.set_index('Date', inplace=True)

    # Агрегация данных по месяцам и переименование колонки
    goog = df['adj_close'].resample('ME').last().rename('adj_close')

    # Разделение данных на обучающую и тестовую выборки
    train_indices = goog.index.year < 2018
    goog_train = goog[train_indices]
    goog_test = goog[~train_indices]
    test_length = len(goog_test)

    # Проверка данных
    print("Обучающая выборка:", goog_train.tail())
    print("Тестовая выборка:", goog_test.head())

    # Модель линейного тренда Хольта
    hs_1 = Holt(goog_train).fit()
    hs_forecast_1 = hs_1.forecast(test_length)

    # Модель экспоненциального тренда Хольта
    hs_2 = Holt(goog_train, exponential=True).fit()
    hs_forecast_2 = hs_2.forecast(test_length)

    # Модель затухающего тренда Хольта
    hs_3 = Holt(goog_train, exponential=False, damped=True).fit(damping_slope=0.99)
    hs_forecast_3 = hs_3.forecast(test_length)

    # Построение графика прогнозов моделей Хольта
    plt.figure(figsize=(12, 6))
    goog.plot(color='blue', title="Модели сглаживания Хольта", label='Фактические данные', legend=True)

    hs_forecast_1.plot(color='orange', legend=True, label='Линейный тренд')
    hs_forecast_2.plot(color='green', legend=True, label='Экспоненциальный тренд')
    hs_forecast_3.plot(color='red', legend=True, label='Затухающий тренд')

    plt.legend()
    plt.show()