import backtrader as bt
import pandas as pd

class SmaSignal(bt.Signal):  # класс для сигнала на основе скользящей средней
    params = (('period', 20),)  # задаем параметр периода для скользящей средней, по умолчанию 20

    def __init__(self):  # инициализируем класс
        # вычисляем сигнал: разница между текущей ценой закрытия и значением SMA
        self.lines.signal = self.data.close - bt.ind.SMA(self.data.close, period=self.p.period)

# читаем данные из CSV файла, парсим даты и устанавливаем 'Date' как индекс
df = pd.read_csv('./AAPL_data.csv', parse_dates=True, index_col='Date')

# создаем объект данных для backtrader на основе загруженного DataFrame
data = bt.feeds.PandasData(dataname=df)

# создаем экземпляр Cerebro, который управляет всей логикой торговли
cerebro = bt.Cerebro(stdstats=False)  # отключаем стандартные статистики

# добавляем данные в Cerebro
cerebro.adddata(data)

# устанавливаем начальный капитал в 1000 долларов
cerebro.broker.setcash(1000.0)

# добавляем наш сигнал на покупку в Cerebro
cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)

# добавляем наблюдатель для отображения сделок (покупок и продаж)
cerebro.addobserver(bt.observers.BuySell)

# добавляем наблюдатель для отображения стоимости портфеля
cerebro.addobserver(bt.observers.Value)

print(f'Начальный портфель: {cerebro.broker.getvalue():.2f}')

# запускаем торговую стратегию
cerebro.run()

print(f'Конечный портфель: {cerebro.broker.getvalue():.2f}')

cerebro.plot(iplot=True, volume=False)