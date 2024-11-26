import pandas as pd
import yfinance as yf
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
import pandas_datareader.data as web

# Настройка графики
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 300
warnings.filterwarnings('ignore')

# Параметры
RISKY_ASSET = 'META'
START_DATE = '2014-01-01'
END_DATE = '2018-12-31'

# Загрузка факторов Фама-Френча
try:
    ff_factors = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=START_DATE)[0]
except Exception as e:
    raise ValueError(f"Ошибка при загрузке факторов Фама-Френча: {e}")

ff_factors = ff_factors / 100  # Преобразование процентов в доли

# Преобразование индекса в datetime и установка даты на конец месяца
if isinstance(ff_factors.index, pd.PeriodIndex):
    ff_factors.index = ff_factors.index.to_timestamp('M')
else:
    ff_factors.index = pd.to_datetime(ff_factors.index).to_period('M').to_timestamp('M')

# Загрузка цен на актив
asset_prices = yf.download(RISKY_ASSET, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
if asset_prices.empty:
    raise ValueError("Не удалось загрузить данные для актива.")

# Расчёт ежемесячных доходностей
asset_returns = asset_prices['Close'].resample('M').last().pct_change().dropna()

# Переименование колонки из 'META' в 'rtn'
if isinstance(asset_returns, pd.Series):
    asset_returns = asset_returns.to_frame('rtn')
elif isinstance(asset_returns, pd.DataFrame):
    asset_returns = asset_returns.rename(columns={RISKY_ASSET: 'rtn'})

# Объединение данных по дате с использованием inner join для совпадающих дат
ff_data = ff_factors.join(asset_returns, how='inner')

# Проверка наличия столбца 'rtn'
if 'rtn' not in ff_data.columns:
    raise KeyError("Столбец 'rtn' отсутствует в объединенных данных. Проверьте согласованность индексов.")

# Расчёт избыточной доходности
ff_data['excess_rtn'] = ff_data['rtn'] - ff_data['RF']

# Оценка трехфакторной модели
# Использование Q() для переменной с дефисом
ff_model = smf.ols(formula='excess_rtn ~ Q("Mkt-RF") + SMB + HML', data=ff_data).fit()

# Сохранение таблиц результатов
# Таблица 1: Общие результаты модели
with open('three_factor_model_summary.txt', 'w') as f:
    f.write(ff_model.summary().as_text())

# Таблица 2: Коэффициенты модели с их статистической значимостью
coefficients = pd.DataFrame({
    'Coefficient': ff_model.params,
    'P-value': ff_model.pvalues
})
coefficients.to_csv('three_factor_coefficients.csv', index=True)

# Вывод таблиц на экран
print("Таблица 1: Результаты оценки трехфакторной модели")
print(ff_model.summary())

print("\nТаблица 2: Коэффициенты модели")
print(coefficients)
