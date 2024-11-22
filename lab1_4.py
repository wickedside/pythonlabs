from datetime import datetime
import backtrader as bt

class RsiSignalStrategy(bt.SignalStrategy):
    params = dict(rsi_periods=14, rsi_upper=70, rsi_lower=30, rsi_mid=50)

    def __init__(self):
        # Инициализация индикатора RSI
        rsi = bt.indicators.RSI(period=self.p.rsi_periods,
                                upperband=self.p.rsi_upper,
                                lowerband=self.p.rsi_lower)

        # Создание сигнала для входа в длинную позицию
        rsi_signal_long = bt.ind.CrossUp(rsi, self.p.rsi_lower)
        self.signal_add(bt.SIGNAL_LONG, rsi_signal_long)

        # Создание сигнала для выхода из длинной позиции
        self.signal_add(bt.SIGNAL_LONGEXIT, -(rsi > self.p.rsi_mid))

        # Создание сигнала для входа в короткую позицию
        rsi_signal_short = bt.ind.CrossDown(rsi, self.p.rsi_upper)
        self.signal_add(bt.SIGNAL_SHORT, rsi_signal_short)

        # Создание сигнала для выхода из короткой позиции
        self.signal_add(bt.SIGNAL_SHORTEXIT, rsi < self.p.rsi_mid)

# Источник данных
data = bt.feeds.YahooFinanceData(dataname='./META_data.csv',
                                  fromdate=datetime(2018, 1, 1),
                                  todate=datetime(2018, 12, 31))

# Инициализация движка Cerebro
cerebro = bt.Cerebro(stdstats=False)
cerebro.addstrategy(RsiSignalStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(1000.0)
cerebro.broker.setcommission(commission=0.001)

# Добавление наблюдателей для покупок/продаж и стоимости портфеля
cerebro.addobserver(bt.observers.BuySell)
cerebro.addobserver(bt.observers.Value)

# Запуск стратегии
cerebro.run()

# Построение результатов
cerebro.plot(iplot=True, volume=False)
