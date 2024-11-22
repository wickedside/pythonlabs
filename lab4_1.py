# Скрипт: task1_garch_modeling.py

import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

# Шаг 1: Загрузка данных
RISKY_ASSET = 'GOOG'
START_DATE = '2015-01-01'
END_DATE = '2018-12-31'

# Исправленный вызов функции download
df = yf.download(RISKY_ASSET, start=START_DATE, end=END_DATE, auto_adjust=True)

# Шаг 2: Расчет дневной доходности
returns = 100 * df['Close'].pct_change().dropna()
returns.name = 'asset_returns'

# Шаг 3: Построение графика доходности
plt.figure(figsize=(10, 6))
returns.plot(title=f'{RISKY_ASSET} Дневная доходность: {START_DATE} - {END_DATE}')
plt.xlabel('Дата')
plt.ylabel('Доходность (%)')
plt.tight_layout()
plt.savefig('task1_returns_plot.png')  # Сохранение изображения
plt.show()

# Шаг 4: Оценка модели ARCH(1)
arch_model_arch = arch_model(returns, mean='Zero', vol='ARCH', p=1, o=0, q=0)
model_arch = arch_model_arch.fit(disp='off')
summary_arch = model_arch.summary()
print(summary_arch)

# Сохранение таблицы с результатами ARCH(1)
with open('task1_arch_summary.txt', 'w') as f:
    f.write(str(summary_arch))

# Шаг 5: Построение графиков остатков и условной волатильности для ARCH(1)
model_arch.plot(annualize='D')
plt.tight_layout()
plt.savefig('task1_arch_plots.png')  # Сохранение изображения
plt.show()

# Шаг 6: Оценка модели GARCH(1,1)
arch_model_garch = arch_model(returns, mean='Zero', vol='GARCH', p=1, o=0, q=1)
model_garch = arch_model_garch.fit(disp='off')
summary_garch = model_garch.summary()
print(summary_garch)

# Сохранение таблицы с результатами GARCH(1,1)
with open('task1_garch_summary.txt', 'w') as f:
    f.write(str(summary_garch))

# Шаг 7: Построение графиков остатков и условной волатильности для GARCH(1,1)
model_garch.plot(annualize='D')
plt.tight_layout()
plt.savefig('task1_garch_plots.png')  # Сохранение изображения
plt.show()
