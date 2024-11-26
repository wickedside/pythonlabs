# Импортируем необходимые библиотеки
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Загружаем ежедневные цены на золото с Yahoo Finance
df = yf.download('GC=F',
                 start='2000-01-01',
                 end='2005-12-31',
                 interval='1d',
                 progress=False)

# Подготавливаем данные
df.reset_index(inplace=True)

# Проверяем названия столбцов до переименования
print("Столбцы до переименования:", df.columns.tolist())

# Проверяем, имеет ли DataFrame MultiIndex столбцов
if isinstance(df.columns, pd.MultiIndex):
    # Если имеет, преобразуем его в одномерный
    df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
    print("Столбцы после преобразования MultiIndex в одномерный:", df.columns.tolist())

# Переименовываем столбцы и выбираем только необходимые
# Теперь столбцы должны быть: ['ds', 'y']
# Используем правильное имя столбца после преобразования MultiIndex
df = df[['Date', 'Adj Close_GC=F']].rename(columns={'Date': 'ds', 'Adj Close_GC=F': 'y'})

# Проверяем названия столбцов после переименования
print("Столбцы после переименования:", df.columns.tolist())

# Проверяем тип данных столбца 'y'
print("Тип данных столбца 'y':", type(df['y']))

# Выводим первые несколько строк столбца 'y'
print("Первые 5 строк столбца 'y':")
print(df['y'].head())

# Удаляем пропущенные значения
df.dropna(inplace=True)

# Проверяем, является ли 'y' Series
if isinstance(df['y'], pd.Series):
    # Преобразуем столбец 'y' в числовой тип
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
else:
    # Если 'y' всё ещё DataFrame, выбираем первый столбец
    df['y'] = pd.to_numeric(df['y'].iloc[:, 0], errors='coerce')

# Удаляем строки с некорректными значениями в 'y'
df.dropna(subset=['y'], inplace=True)

# Разделяем данные на обучающую и тестовую выборки
df['ds'] = pd.to_datetime(df['ds'])
train_indices = df['ds'].dt.year < 2005
df_train = df.loc[train_indices].reset_index(drop=True)
df_test = df.loc[~train_indices].reset_index(drop=True)

# Проверка структуры обучающей выборки
print("\nОбучающая выборка:")
print(df_train.head())
print(df_train.dtypes)

# Создаем и обучаем модель Prophet
model_prophet = Prophet(seasonality_mode='additive')
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_prophet.fit(df_train)

# Прогнозируем цены на золото
df_future = model_prophet.make_future_dataframe(periods=365)
df_forecast = model_prophet.predict(df_future)

# Строим прогноз
fig1 = model_prophet.plot(df_forecast)
plt.title('Forecasted Gold Prices')
plt.show()

# Проверяем разложение временного ряда
fig2 = model_prophet.plot_components(df_forecast)
plt.show()

# Объединяем тестовый набор с прогнозами
df_pred = df_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
df_compare = pd.merge(df_test, df_pred, on='ds', how='left')
df_compare.set_index('ds', inplace=True)

# Строим график фактических и прогнозных цен
plt.figure(figsize=(12,6))
plt.plot(df_compare.index, df_compare['y'], label='Actual', color='blue')
plt.plot(df_compare.index, df_compare['yhat'], label='Forecast', color='orange')
plt.fill_between(df_compare.index, df_compare['yhat_lower'], df_compare['yhat_upper'], color='lightblue', alpha=0.5)
plt.title('Gold Price - Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Gold Price ($)')
plt.legend()
plt.show()
