# Скрипт: task2_ccc_garch.py

import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Шаг 1: Загрузка данных
RISKY_ASSETS = ['GOOG', 'MSFT', 'AAPL']
START_DATE = '2015-01-01'
END_DATE = '2018-12-31'

# Исправленный вызов функции download
df = yf.download(RISKY_ASSETS, start=START_DATE, end=END_DATE, auto_adjust=True)

# Проверка структуры загруженных данных
print("Структура данных после загрузки:")
print(df.head())
print("\nДоступные столбцы:")
print(df.columns)

# Шаг 2: Расчет дневной доходности
# При auto_adjust=True столбец 'Close' уже скорректирован
returns = 100 * df['Close'].pct_change().dropna()
returns.columns = RISKY_ASSETS

# Проверка расчетных доходностей
print("\nПервые пять строк доходностей:")
print(returns.head())

# Шаг 3: Построение графиков доходности
returns.plot(subplots=True, figsize=(10, 8), title=f'Дневная доходность акций: {START_DATE} - {END_DATE}')
plt.xlabel('Дата')
plt.ylabel('Доходность (%)')
plt.tight_layout()
plt.savefig('task2_returns_plots.png')  # Сохранение изображения
plt.show()

# Шаг 4: Оценка одномерных моделей GARCH(1,1)
coeffs = []
cond_vol = []
std_resids = []
models = []

for asset in RISKY_ASSETS:
    print(f"\nОценка модели GARCH(1,1) для {asset}...")
    model = arch_model(returns[asset], mean='Constant', vol='GARCH', p=1, o=0, q=1).fit(disp='off')
    coeffs.append(model.params)
    cond_vol.append(model.conditional_volatility)
    std_resids.append(model.resid / model.conditional_volatility)
    models.append(model)
    print(f"Модель для {asset} успешно оценена.")

# Шаг 5: Создание таблиц с результатами
coeffs_df = pd.DataFrame(coeffs, index=RISKY_ASSETS)
cond_vol_df = pd.DataFrame(cond_vol).transpose()
cond_vol_df.columns = RISKY_ASSETS
std_resids_df = pd.DataFrame(std_resids).transpose()
std_resids_df.columns = RISKY_ASSETS

# Сохранение таблиц
coeffs_df.to_csv('task2_coefficients.csv')
cond_vol_df.to_csv('task2_conditional_volatility.csv')
std_resids_df.to_csv('task2_standardized_residuals.csv')

print("\nТаблицы с результатами успешно сохранены:")
print(" - task2_coefficients.csv")
print(" - task2_conditional_volatility.csv")
print(" - task2_standardized_residuals.csv")

# Шаг 6: Вычисление постоянной матрицы условной корреляции (R)
R = std_resids_df.corr()

# Сохранение матрицы корреляции
R.to_csv('task2_correlation_matrix.csv')
print("\nПостоянная матрица условной корреляции (R) сохранена как task2_correlation_matrix.csv")

# Шаг 7: Прогнозирование условной ковариационной матрицы на один шаг вперед
diag = []
N = len(RISKY_ASSETS)
D = np.zeros((N, N))

print("\nПрогнозирование условной ковариационной матрицы на один шаг вперед...")
for model in models:
    forecast = model.forecast(horizon=1)
    variance_forecast = forecast.variance.values[-1, 0]
    diag.append(np.sqrt(variance_forecast))

diag = np.array(diag)
np.fill_diagonal(D, diag)
H = D @ R.values @ D

# Создание DataFrame для матрицы ковариаций
H_df = pd.DataFrame(H, index=RISKY_ASSETS, columns=RISKY_ASSETS)
H_df.to_csv('task2_conditional_covariance_forecast.csv')
print("Прогноз условной ковариационной матрицы сохранен как task2_conditional_covariance_forecast.csv")

# Шаг 8: Построение итогового изображения (тепловая карта матрицы корреляции)
plt.figure(figsize=(8, 6))
sns.heatmap(R, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Постоянная матрица условной корреляции (CCC-GARCH)')
plt.tight_layout()
plt.savefig('task2_correlation_heatmap.png')  # Сохранение изображения
plt.show()

print("\nТепловая карта матрицы корреляции сохранена как task2_correlation_heatmap.png")
