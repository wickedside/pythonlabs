import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.formula.api as smf

# Определение параметров
RISKY_ASSET = 'META'
START_DATE = '2013-12-31'
END_DATE = '2018-12-31'

# Загрузка данных Fama-French из FRED (Federal Reserve Economic Data)
ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=START_DATE, end=END_DATE)[0]

# Переименование столбцов
ff_data.columns = ['mkt', 'smb', 'hml', 'rf']

# Преобразование значений в числовые и деление на 100
ff_data = ff_data.div(100)

# Скачивание цен на рискованный актив
asset_df = yf.download(RISKY_ASSET, start=START_DATE, end=END_DATE)

# Расчёт ежемесячной доходности рискованного актива
if 'Adj Close' in asset_df.columns:
    y = asset_df['Adj Close'].resample('M').last().pct_change().dropna()
    y.index = y.index.to_period('M')  # Устанавливаем тот же формат индекса
    y.name = 'rtn'
else:
    print("Столбец 'Adj Close' отсутствует в данных для рискованного актива.")
    exit()

# Проверка данных перед объединением
print("\nДанные Fama-French (первые строки):")
print(ff_data.head())
print("\nДанные доходности рискованного актива (первые строки):")
print(y.head())

# Объединение наборов данных
ff_data = ff_data.join(y, how='left')

# Переименование столбца после объединения
ff_data.rename(columns={RISKY_ASSET: 'rtn'}, inplace=True)

# Проверка данных после объединения
print("\nДанные после объединения (первые строки):")
print(ff_data.head())

# Проверка наличия данных перед оценкой модели
if 'rtn' in ff_data.columns and not ff_data['rtn'].isnull().all():
    ff_data['excess_rtn'] = ff_data['rtn'] - ff_data['rf']

    # Проверка, есть ли ненулевые данные для регрессии
    if not ff_data['excess_rtn'].isnull().all():
        # Оценка трехфакторной модели
        ff_model = smf.ols(formula='excess_rtn ~ mkt + smb + hml', data=ff_data).fit()
        print(ff_model.summary())
    else:
        print("Нет данных для оценки модели: все значения 'excess_rtn' равны NaN.")
else:
    print("Колонка 'rtn' отсутствует или все значения равны NaN.")
