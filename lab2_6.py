import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Загрузка данных
goog = pd.read_csv('./google_data.csv')

# Преобразование даты в формат datetime и установка ее в качестве индекса
goog['Date'] = pd.to_datetime(goog['Date'])
goog.set_index('Date', inplace=True)

# Выбор только столбца с ценами акций
prices = goog['Close']

# Создание модели ARIMA(2,1,1)
model_arima = ARIMA(prices, order=(2, 1, 1))
arima_fit = model_arima.fit()

# Прогнозирование на первые три месяца 2019 года (12 недель)
arima_forecast = arima_fit.forecast(steps=12)

# Создание временного индекса для прогнозов
forecast_index = pd.date_range(start='2019-01-01', periods=12, freq='W')

# Создание графиков
fig, ax = plt.subplots(figsize=(12, 6))

# Исторические данные
ax.plot(prices, label='Historical Prices', color='blue')

# Прогноз ARIMA
ax.plot(forecast_index, arima_forecast, label='ARIMA(2,1,1) Forecast', color='red', marker='o')

# Оформление графика
ax.set_title('Google Stock Prices and Forecast (ARIMA)')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid()

# Отображение графика
plt.tight_layout()
plt.show()
