import pandas as pd
import seaborn as sns
from prophet import Prophet  # Используем новую библиотеку
import matplotlib.pyplot as plt

# Загрузка данных из CSV файла
csv_file_path = r'./gold_prices.csv'
df = pd.read_csv(csv_file_path, skiprows=2)

# Проверка текущих столбцов
print("Столбцы после пропуска строк:", df.columns)

# Используем первый столбец как 'ds' и второй числовой столбец как 'y'
df.rename(columns={'Date': 'ds', 'Unnamed: 1': 'y'}, inplace=True)

# Преобразование столбца 'ds' в формат даты
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# Удаление строк с некорректными датами
df = df.dropna(subset=['ds', 'y'])

# Преобразование столбца 'y' в числовой формат
df['y'] = pd.to_numeric(df['y'], errors='coerce')

# Удаление строк с некорректными значениями 'y'
df = df.dropna(subset=['y'])

# Оставляем только нужные столбцы
df = df[['ds', 'y']]

# Проверка данных после обработки
print(df.head())

# Фильтрация данных по дате
start_date = '2000-01-01'
end_date = '2005-12-31'
df_filtered = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

# Проверка, что данные не пустые
if df_filtered.empty:
    print("Нет данных в указанных пределах дат.")
else:
    # Создание и обучение модели Prophet
    model_prophet = Prophet(seasonality_mode='additive')
    model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model_prophet.fit(df_filtered)

    # Создание будущих дат для предсказания
    df_future = model_prophet.make_future_dataframe(periods=365)
    df_pred = model_prophet.predict(df_future)

    # Выбор нужных колонок
    selected_columns = ['ds', 'yhat_lower', 'yhat_upper', 'yhat']
    df_pred = df_pred.loc[:, selected_columns].reset_index(drop=True)

    # Подготовка тестовой выборки для объединения с предсказаниями
    df_test = df_filtered.copy()  # Используем отфильтрованные данные как тестовые
    df_test = df_test.merge(df_pred, on=['ds'], how='left')

    # Установка индекса по дате
    df_test.set_index('ds', inplace=True)

    # Визуализация фактических и предсказанных цен на золото
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax = sns.lineplot(data=df_test[['y', 'yhat_lower', 'yhat_upper', 'yhat']])

    ax.fill_between(df_test.index,
                    df_test['yhat_lower'],
                    df_test['yhat_upper'],
                    alpha=0.3)

    ax.set(title='Gold Price - Actual vs. Predicted',
           xlabel='Date',
           ylabel='Gold Price ($)')

    plt.show()
