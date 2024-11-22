import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# Загрузка данных с пропуском первых двух строк
file_path = r'./gold_prices.csv'
df = pd.read_csv(file_path, skiprows=2)

# Выводим текущие столбцы для отладки
print("Столбцы после пропуска строк:", df.columns)

# Вывод первых строк данных
print("Первые строки данных:")
print(df.head())

# Указываем, какие столбцы использовать
if 'Date' in df.columns and 'Unnamed: 1' in df.columns:
    df.rename(columns={'Date': 'Date', 'Unnamed: 1': 'Value'}, inplace=True)
else:
    print("Ошибка: Ожидаемые столбцы 'Date' и/или 'Unnamed: 1' отсутствуют.")
    exit()

# Преобразование столбца 'Date' в формат datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Преобразование столбца 'Value' в числовой формат
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

# Удаление строк с некорректными значениями в 'Date' и 'Value'
df = df.dropna(subset=['Date', 'Value'])

# Установим 'Date' как индекс
df.set_index('Date', inplace=True)

# Проверка данных после обработки
print("Первые строки данных после обработки:")
print(df.head())

# Тест Дики-Фуллера (ADF)
def adf_test(x):
    indices = ['Статистика теста', 'p-значение', 'Число использованных лагов', 'Число наблюдений']
    adf_test = adfuller(x, autolag='AIC')
    results = pd.Series(adf_test[0:4], index=indices)
    for key, value in adf_test[4].items():
        results[f'Критическое значение ({key})'] = value
    return results

# Выполнение теста ADF на данных о ценах
print("Результаты теста ADF:")
print(adf_test(df['Value']))

# Тест Квятковского-Филлипса-Шмидта-Шина (KPSS)
def kpss_test(x, h0_type='c'):
    indices = ['Статистика теста', 'p-значение', 'Число лагов']
    kpss_test = kpss(x, regression=h0_type)
    results = pd.Series(kpss_test[0:3], index=indices)
    for key, value in kpss_test[3].items():
        results[f'Критическое значение ({key})'] = value
    return results

# Выполнение теста KPSS на данных о ценах
print("\nРезультаты теста KPSS:")
print(kpss_test(df['Value']))

# Графики автокорреляции и частичной автокорреляции
N_LAGS = 40
SIGNIFICANCE_LEVEL = 0.05

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df['Value'], ax=ax[0], lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL)
ax[0].set_title('Функция автокорреляции (ACF)')

plot_pacf(df['Value'], ax=ax[1], lags=N_LAGS, alpha=SIGNIFICANCE_LEVEL)
ax[1].set_title('Функция частичной автокорреляции (PACF)')

plt.tight_layout()
plt.show()
