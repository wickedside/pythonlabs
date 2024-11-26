# lab3_3.py
# Задание 3: Реализация скользящей трехфакторной модели на портфеле активов

import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import warnings

plt.style.use('seaborn-v0_8')


plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['figure.dpi'] = 300
warnings.simplefilter(action='ignore', category=FutureWarning)

# Параметры
ASSETS = ['AMZN', 'GOOG', 'AAPL', 'MSFT']
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
START_DATE = '2010-01-01'
END_DATE = '2018-12-31'

# Загрузка трехфакторных данных Фама-Френча
print("Загрузка трехфакторных данных Фама-Френча...")
df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=START_DATE)[0]
df_three_factor = df_three_factor.div(100)
df_three_factor.index = df_three_factor.index.strftime('%Y-%m')

# Загрузка цен на активы с корректным параметром auto_adjust
print("Загрузка цен на активы...")
asset_df = yf.download(ASSETS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Проверка наличия данных
if asset_df.empty:
    raise ValueError("Не удалось загрузить данные для активов.")

# Расчет ежемесячной доходности
asset_df = asset_df['Close'].resample('M').last().pct_change().dropna()
asset_df.index = asset_df.index.strftime('%Y-%m')

# Проверка на наличие всех активов в данных
if not all(asset in asset_df.columns for asset in ASSETS):
    missing_assets = [asset for asset in ASSETS if asset not in asset_df.columns]
    raise ValueError(f"Отсутствуют данные для активов: {missing_assets}")

# Расчет доходности портфеля
asset_df['portfolio_returns'] = asset_df[ASSETS].dot(WEIGHTS)

# Объединение данных
ff_data = asset_df.join(df_three_factor).drop(ASSETS, axis=1)
ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
ff_data['portf_ex_rtn'] = ff_data['portf_rtn'] - ff_data['rf']

# Проверка наличия необходимых столбцов
required_columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
if not all(col in ff_data.columns for col in required_columns):
    raise ValueError(f"Некоторые необходимые столбцы отсутствуют в данных: {required_columns}")

# Функция для скользящей модели
def rolling_factor_model(input_data, formula, window_size):
    coeffs = []
    for start in range(len(input_data) - window_size + 1):
        end = start + window_size
        window_data = input_data.iloc[start:end]
        model = smf.ols(formula=formula, data=window_data).fit()
        coeffs.append(model.params)
    coeffs_df = pd.DataFrame(coeffs, index=input_data.index[window_size-1:])
    return coeffs_df

# Параметры модели
MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
WINDOW_SIZE = 60  # 60 месяцев = 5 лет

# Оценка скользящей модели
print("Оценка скользящей трехфакторной модели...")
results_df = rolling_factor_model(ff_data, MODEL_FORMULA, WINDOW_SIZE)

# Проверка наличия результатов
if results_df.empty:
    raise ValueError("Результаты скользящей модели пусты. Проверьте параметры окна и данные.")

# Визуализация результатов
plt.figure(figsize=(14,8))
plt.plot(results_df.index, results_df['mkt'], label='Коэффициент MKT')
plt.plot(results_df.index, results_df['smb'], label='Коэффициент SMB')
plt.plot(results_df.index, results_df['hml'], label='Коэффициент HML')
plt.xlabel('Дата')
plt.ylabel('Значение коэффициента')
plt.title('Скользящая трехфакторная модель Фама-Френча')
plt.legend()
plt.savefig('rolling_fama_french_three_factor.png')
plt.show()
