# Импорт необходимых библиотек
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import pandas_datareader.data as web
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import seaborn as sns

# Установка стиля графиков
sns.set_palette('cubehelix')

# Шаг 1: Загрузка данных цен на золото (используем ETF GLD как замену)
def load_gold_data(start_date, end_date):
    df = yf.download('GLD', start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError("Не удалось загрузить данные GLD. Проверьте правильность тикера и доступность данных.")

    # Проверка и преобразование MultiIndex колонок
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.rename(columns={'Adj Close': 'price'}, inplace=True)
    return df

# Шаг 2: Загрузка данных CPI из FRED
def load_cpi_data(start_date, end_date):
    try:
        cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
        cpi.rename(columns={'CPIAUCSL': 'CPI'}, inplace=True)
        return cpi
    except Exception as e:
        print(f"Ошибка при загрузке данных CPI: {e}")
        return None

# Шаг 3: Объединение данных GLD и CPI
def merge_data(gold_df, cpi_df):
    # Ресэмплинг данных на ежемесячную частоту
    gold_monthly = gold_df.resample('MS').last()
    print("Ресэмплинг gold_df завершён.")

    cpi_monthly = cpi_df.resample('MS').last()
    print("Ресэмплинг cpi_df завершён.")

    # Установка одинаковых имён индексов
    gold_monthly.index.name = 'Date'
    cpi_monthly.index.name = 'Date'

    # Сброс индексов
    gold_monthly = gold_monthly.reset_index()
    cpi_monthly = cpi_monthly.reset_index()

    # Дополнительная отладка после сброса индексов
    print(f"После сброса индексов - Колонки GLD: {gold_monthly.columns}", flush=True)
    print(f"После сброса индексов - Колонки CPI: {cpi_monthly.columns}", flush=True)

    # Установка индекса 'Date'
    gold_monthly.set_index('Date', inplace=True)
    cpi_monthly.set_index('Date', inplace=True)

    # Объединение по индексу
    merged_df = pd.merge(gold_monthly, cpi_monthly, left_index=True, right_index=True, how='inner')
    print("Объединение данных завершено.")

    return merged_df

# Шаг 4: Дефляция цен на золото к значению на дату 2011-12-31
def deflate_prices(df, defl_date):
    # Преобразуем строковую дату в datetime
    defl_date = pd.to_datetime(defl_date)

    # Индекс CPI на дату дефляции
    try:
        cpi_def = df.loc[defl_date, 'CPI']
    except KeyError:
        # Если точная дата отсутствует, используем ближайшую предыдущую дату
        cpi_def = df['CPI'].asof(defl_date)
        print(f"Точная дата CPI для {defl_date.date()} отсутствует. Использована ближайшая доступная дата: {cpi_def}")

    # Дефляция цен
    df['price_deflated'] = df['price'] * (cpi_def / df['CPI'])

    return df

# Шаг 5: Применение натурального логарифма и расчет скользящих показателей
def compute_log_and_rolling(df, window):
    df['price_log'] = np.log(df['price_deflated'])
    df['rolling_mean_log'] = df['price_log'].rolling(window=window).mean()
    df['rolling_std_log'] = df['price_log'].rolling(window=window).std()
    return df

# Шаг 6: Проверка на стационарность
def test_autocorrelation(x, lags=40, title=''):
    print(f"\nТест стационарности для: {title}")
    # Тест Дики-Фуллера
    adf_result = adfuller(x.dropna())
    print('ADF Statistic: %f' % adf_result[0])
    print('p-value: %f' % adf_result[1])

    # Тест KPSS
    try:
        kpss_result = kpss(x.dropna(), regression='c', nlags='auto')
        print('KPSS Statistic: %f' % kpss_result[0])
        print('p-value: %f' % kpss_result[1])
    except Exception as e:
        print(f"Ошибка при выполнении теста KPSS: {e}")

    # Построение графиков ACF и PACF
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(x.dropna(), lags=lags, ax=ax[0])
    plot_pacf(x.dropna(), lags=lags, ax=ax[1])
    plt.tight_layout()
    plt.show()

# Шаг 7: Основной процесс
def main():
    # Определение периодов
    start_date = '2000-01-01'
    end_date = '2011-12-31'
    defl_date = '2011-12-31'
    window = 12

    # Загрузка данных
    print("Загрузка данных GLD...")
    gold_df = load_gold_data(start_date, end_date)
    print("Загрузка данных CPI...")
    cpi_df = load_cpi_data(start_date, end_date)

    if cpi_df is None:
        print("Не удалось загрузить данные CPI. Прерывание выполнения.")
        return

    # Объединение данных
    print("Объединение данных GLD и CPI...")
    merged_df = merge_data(gold_df, cpi_df)

    # Дефляция цен
    print("Корректировка цен с учетом инфляции...")
    merged_df = deflate_prices(merged_df, defl_date)

    # Проверка на наличие NaN в price_deflated
    if merged_df['price_deflated'].isnull().any():
        print("Некоторые значения price_deflated равны NaN. Проверьте корректность данных.")
    else:
        print("Все значения price_deflated корректны.")

    # График цен до и после дефляции
    merged_df[['price', 'price_deflated']].plot(title='Цена золота (дефлированная)')
    plt.xlabel('Дата')
    plt.ylabel('Цена (USD)')
    plt.show()

    # Логарифмическое преобразование и скользящие показатели
    print("Применение логарифмического преобразования и расчет скользящих показателей...")
    merged_df = compute_log_and_rolling(merged_df, window)

    # График логарифмированных цен и скользящих показателей
    merged_df[['price_log', 'rolling_mean_log', 'rolling_std_log']].plot(
        title='Логарифм цен золота и скользящие показатели')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.show()

    # Проверка стационарности логарифмированных цен
    print("Проверка стационарности логарифмированных цен:")
    test_autocorrelation(merged_df['price_log'], lags=40, title='Логарифмированные цены (до разностей)')

    # Применение разности первого порядка и расчет скользящих показателей
    print("Применение разности первого порядка и расчет скользящих показателей...")
    merged_df['price_log_diff'] = merged_df['price_log'].diff(1)
    merged_df['roll_mean_log_diff'] = merged_df['price_log_diff'].rolling(window).mean()
    merged_df['roll_std_log_diff'] = merged_df['price_log_diff'].rolling(window).std()

    # График разностей первого порядка и скользящих показателей
    merged_df[['price_log_diff', 'roll_mean_log_diff', 'roll_std_log_diff']].plot(
        title='Цена золота (разности первого порядка)')
    plt.xlabel('Дата')
    plt.ylabel('Значение')
    plt.show()

    # Проверка стационарности после разностей
    #print("Проверка стационарности после разностей первого порядка:")
    #test_autocorrelation(merged_df['price_log_diff'], lags=40,
    #                     title='Логарифмированные цены (после разностей первого порядка)')


if __name__ == "__main__":
    main()
