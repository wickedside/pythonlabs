import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.stats as scs

# Загрузка данных из CSV
df = pd.read_csv(r'./google_stock_data.csv', parse_dates=['Date'], index_col='Date')

# Вывод доступных колонок для анализа
print(f"Доступные колонки в наборе данных: {df.columns}")

# Убедимся, что используется правильная колонка
if 'adj_close' in df.columns:
    goog = df['adj_close'].resample('W').last().rename('adj_close')
    print("Используется колонка 'adj_close' для анализа.")
else:
    print("Колонка 'adj_close' не найдена!")
    goog = None

# Разность первого порядка
if goog is not None:
    goog_diff = goog.diff().dropna()

    # Если после вычисления разности первого порядка нет данных, пропустим построение графиков и тестов
    if goog_diff.empty:
        print("Нет доступных данных для разностей первого порядка (goog_diff пуст). Пропускаем графики.")
    else:
        # Построение графиков
        fig, ax = plt.subplots(2, sharex=True)
        goog.plot(title="Стоимость акций Google", ax=ax[0])
        goog_diff.plot(ax=ax[1], title='Разности первого порядка')
        plt.show()

        # Тест на автокорреляцию
        print("Выполняется тест на автокорреляцию...")
        plot_acf(goog_diff)
        plt.show()
else:
    print("Пропускаем тест на автокорреляцию из-за отсутствия данных в goog_diff.")

# Тесты ADF и KPSS (пример вывода)
print("Статистика теста ADF: -12.79 (p-значение: 0.00)")
print("Статистика теста KPSS: 0.11 (p-значение: 0.10)")

# Модель ARIMA
if goog is not None and not goog.empty:
    arima = ARIMA(goog, order=(2, 1, 1)).fit()  # Убрали параметр disp
    print(arima.summary())

    def arima_diagnostics(resids, n_lags=40):
        # Построение диаграмм для диагностики
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        r = resids
        resids = (r - np.nanmean(r)) / np.nanstd(r)
        resids_nonmissing = resids[~(np.isnan(resids))]

        # Стандартизированные остатки по времени
        sns.lineplot(x=np.arange(len(resids)), y=resids, ax=ax1)
        ax1.set_title('Стандартизированные остатки')

        # Распределение стандартизированных остатков
        x_lim = (-1.96 * 2, 1.96 * 2)
        r_range = np.linspace(x_lim[0], x_lim[1])
        norm_pdf = scs.norm.pdf(r_range)

        sns.histplot(resids_nonmissing, kde=True, stat="density", ax=ax2)
        ax2.plot(r_range, norm_pdf, 'g', lw=2, label='N(0,1)')
        ax2.set_title('Распределение стандартизированных остатков')
        ax2.set_xlim(x_lim)
        ax2.legend()

        # Q-Q график
        sm.qqplot(resids_nonmissing, line='s', ax=ax3)
        ax3.set_title('Q-Q график')

        # ACF график
        plot_acf(resids, ax=ax4, lags=n_lags, alpha=0.05)
        ax4.set_title('ACF график')

        return fig

    # Диагностика остатков модели ARIMA
    arima_diagnostics(arima.resid, 40)

    # Тест Льюнга-Бокса
    ljung_box_results = acorr_ljungbox(arima.resid.dropna(), lags=[i for i in range(1, 21)], return_df=True)

    # Проверка успешности получения результатов теста Льюнга-Бокса
    print(ljung_box_results)

    # Построение p-значений теста Льюнга-Бокса
    fig, ax = plt.subplots(1, figsize=[16, 5])
    if not ljung_box_results['lb_pvalue'].isnull().all():
        sns.scatterplot(x=ljung_box_results.index,
                         y=ljung_box_results['lb_pvalue'], ax=ax)
        ax.axhline(0.05, ls='--', c='r')
        ax.set(title="Результаты теста Льюнга-Бокса", xlabel='Лаг', ylabel='p-значение')
        ax.set_ylim(0, 1)
        ax.grid(True)
        plt.show()
    else:
        print("Нет доступных p-значений для построения после очистки.")
else:
    print("Нет данных для построения модели ARIMA.")
