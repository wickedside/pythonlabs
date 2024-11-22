import backtrader as bt
import datetime

import pandas as pd
import yfinance as yf

# Загрузка данных MSFT через yfinance
data = yf.download("MSFT", start="2018-01-01", end="2018-12-31")

# Сохранение данных в CSV
csv_file_path = './MSFT_data.csv'
data.to_csv(csv_file_path)

# Обработка CSV-файла: пропуск первых строк и переименование столбцов
data = pd.read_csv(csv_file_path, skiprows=2)  # Пропускаем первые две строки
data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']  # Задаем корректные заголовки
data['Date'] = pd.to_datetime(data['Date'])  # Преобразуем столбец 'Date' в формат даты
data.to_csv(csv_file_path, index=False)  # Сохраняем обработанный файл

# Определение стратегии
class BBand_Strategy(bt.Strategy):
    params = (('period', 20), ('devfactor', 2.0),)

    def __init__(self):
        # Отслеживание цены закрытия в серии
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        # Отслеживание отложенных ордеров / цены покупки / комиссии за покупку
        self.order = None
        self.price = None
        self.comm = None

        # Индикатор полос Боллинджера
        self.b_band = bt.ind.BollingerBands(self.datas[0], period=self.p.period, devfactor=self.p.devfactor)

        # Сигналы на покупку и продажу
        self.buy_signal = bt.ind.CrossOver(self.datas[0], self.b_band.lines.bot)
        self.sell_signal = bt.ind.CrossOver(self.datas[0], self.b_band.lines.top)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Failed')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    def next(self):
        if not self.position:
            if self.buy_signal > 0:
                size = int(self.broker.getcash() / self.data_open[0])
                self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                self.buy(size=size)
        else:
            if self.sell_signal < 0:
                self.log(f'SELL CREATED --- Size: {self.position.size}')
                self.sell(size=self.position.size)

# Загрузка данных из локального CSV-файла
data = bt.feeds.GenericCSVData(
    dataname=csv_file_path,
    dtformat='%Y-%m-%d',
    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=6,
    openinterest=-1  # Если нет столбца с открытым интересом
)

# Создание экземпляра Cerebro
cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
cerebro.addstrategy(BBand_Strategy)
cerebro.adddata(data)
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
backtest_result = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
cerebro.plot(iplot=True, volume=False)