import yfinance as yf
import matplotlib.pyplot as plt

# Загрузка данных о золоте (например, тикер GLD)
df = yf.download('GLD', start='2000-01-01', end='2011-12-31')

# Переименование колонки для удобства
df.rename(columns={'Close': 'price'}, inplace=True)

# Расчет скользящего среднего и стандартного отклонения
WINDOW_SIZE = 12
df['rolling_mean'] = df['price'].rolling(window=WINDOW_SIZE).mean()
df['rolling_std'] = df['price'].rolling(window=WINDOW_SIZE).std()

# Сохранение данных в CSV файл
csv_file_path = 'gold_prices.csv'
df.to_csv(csv_file_path)

# Визуализация данных
plt.figure(figsize=(14, 7))
plt.plot(df['price'], label='Цена золота', color='gold')
plt.plot(df['rolling_mean'], label='Скользящее среднее', color='blue')
plt.fill_between(df.index, df['rolling_mean'] - df['rolling_std'], df['rolling_mean'] + df['rolling_std'], 
                 color='lightblue', alpha=0.5, label='Стандартное отклонение')
plt.title('Цена золота с скользящим средним и стандартным отклонением')
plt.xlabel('Дата')
plt.ylabel('Цена (USD)')
plt.legend()
plt.show()

print(f"Данные успешно сохранены в {csv_file_path}")