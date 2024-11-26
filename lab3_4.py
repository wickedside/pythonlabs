import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
import pandas_datareader.data as web

# Параметры
RISKY_ASSET = 'AMZN'
START_DATE = '2013-12-31'
END_DATE = '2018-12-31'

# Получение данных трехфакторной модели
df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=START_DATE)[0]
df_three_factor.index = df_three_factor.index.astype(str)  # Преобразование индекса в строку

# Получение данных фактора моментума
df_mom = web.DataReader('F-F_Momentum_Factor', 'famafrench', start=START_DATE)[0]
df_mom.index = df_mom.index.astype(str)  # Преобразование индекса в строку

# Получение данных пятфакторной модели
df_five_factor = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start=START_DATE)[0]
df_five_factor.index = df_five_factor.index.astype(str)  # Преобразование индекса в строку

# Загрузка данных рискованного актива
asset_df = yf.download(RISKY_ASSET, start=START_DATE, end=END_DATE, progress=False)

# Вычисление месячной доходности для рискованного актива
y = asset_df['Adj Close'].resample('M').last().pct_change().dropna()
y.index = y.index.strftime('%Y-%m')  # Форматирование индекса как строки
y.name = 'return'  # Название серии доходности

# Объединение всех наборов данных
four_factor_data = df_three_factor.join(df_mom).join(y)

# Переименование столбцов
four_factor_data.columns = ['mkt', 'smb', 'hml', 'rf', 'mom', 'rtn']

# Деление всех столбцов, кроме 'rtn', на 100
four_factor_data.loc[:, four_factor_data.columns != 'rtn'] /= 100

# Выбор нужного периода
four_factor_data = four_factor_data.loc[START_DATE:END_DATE]

# Вычисление избыточной доходности
four_factor_data['excess_rtn'] = four_factor_data.rtn - four_factor_data.rf

# Объединение данных пятфакторной модели
five_factor_data = df_five_factor.join(y)

# Переименование столбцов
five_factor_data.columns = ['mkt', 'smb', 'hml', 'rmw', 'cma', 'rf', 'rtn']

# Деление всех столбцов, кроме 'rtn', на 100
five_factor_data.loc[:, five_factor_data.columns != 'rtn'] /= 100

# Выбор нужного периода
five_factor_data = five_factor_data.loc[START_DATE:END_DATE]

# Вычисление избыточной доходности
five_factor_data['excess_rtn'] = five_factor_data.rtn - five_factor_data.rf

# Построение четырехфакторной модели
four_factor_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml + mom', data=four_factor_data).fit()
print("Сводка четырехфакторной модели:")
print(four_factor_model.summary())

# Построение пятфакторной модели
five_factor_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml + rmw + cma', data=five_factor_data).fit()
print("\nСводка пятфакторной модели:")
print(five_factor_model.summary())
