# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web
import matplotlib.pyplot as plt

# Определение параметров
ASSETS = ['AMZN', 'GOOG', 'AAPL', 'MSFT']
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
START_DATE = '2009-12-31'
END_DATE = '2018-12-31'

# Загрузка данных о факторах
df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=START_DATE)[0]
df_three_factor = df_three_factor.div(100)

# Изменение формата индекса
df_three_factor.index = df_three_factor.index.astype(str)

# Загрузка цен на рискованные активы из Yahoo Finance
asset_df = yf.download(ASSETS, start=START_DATE, end=END_DATE, progress=False)
print(f'Скачано {asset_df.shape[0]} данных.')

# Расчет ежемесячной доходности рискованных активов
asset_df = asset_df['Adj Close'].resample('M').last().pct_change().dropna()
asset_df.index = asset_df.index.strftime('%Y-%m')

# Расчёт доходности портфеля
asset_df['portfolio_returns'] = np.matmul(asset_df[ASSETS].values, WEIGHTS)

# Объединение наборов данных
ff_data = asset_df.join(df_three_factor).drop(ASSETS, axis=1)
ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
ff_data['portf_ex_rtn'] = ff_data.portf_rtn - ff_data.rf

# Определение функции для скользящей n-факторной модели
def rolling_factor_model(input_data, formula, window_size):
    coeffs = []
    for start_index in range(len(input_data) - window_size + 1):
        end_index = start_index + window_size
        ff_model = smf.ols(formula=formula, data=input_data[start_index:end_index]).fit()
        coeffs.append(ff_model.params)
    coeffs_df = pd.DataFrame(coeffs, index=input_data.index[window_size - 1:])
    return coeffs_df

# Оценка скользящей трехфакторной модели
MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
results_df = rolling_factor_model(ff_data, MODEL_FORMULA, window_size=60)

# Вывод результатов на график
results_df.plot(title='Rolling Fama-French Three-Factor Model')
plt.xlabel('Date')
plt.ylabel('Coefficient Value')
plt.show()