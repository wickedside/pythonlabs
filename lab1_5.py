import sys
import io
import pandas as pd
import cufflinks as cf
from plotly.offline import plot
import tkinter as tk
from tkinter import ttk, messagebox


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

files = ['AAPL_data.csv', 'META_data.csv', 'MSFT_data.csv']
indicators = ['Bollinger Bands', 'MACD', 'RSI']

def ta_dashboard(file, indicator, bb_k, bb_n, macd_fast, macd_slow, macd_signal, rsi_periods, rsi_upper, rsi_lower):
    try:
        
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        
        qf = cf.QuantFig(df, title=f'TA Dashboard - {file}', legend='right', name=f'{file}')
        
        # Add indicators based on user selection
        if 'Bollinger Bands' in indicator:
            qf.add_bollinger_bands(periods=bb_n, boll_std=bb_k)
        if 'MACD' in indicator:
            qf.add_macd(fast_period=macd_fast, slow_period=macd_slow, signal_period=macd_signal)
        if 'RSI' in indicator:
            qf.add_rsi(periods=rsi_periods, rsi_upper=rsi_upper, rsi_lower=rsi_lower, showbands=True)
        
        
        plot(qf.iplot(asFigure=True), filename='ta_dashboard.html', auto_open=True)

    except Exception as e:
        messagebox.showerror("Error", f"Error processing file {file}: {e}")

def on_submit():
    file = file_combobox.get()
    indicator = [indicators[i] for i in indicator_listbox.curselection()]
    bb_k = bb_k_slider.get()
    bb_n = bb_n_slider.get()
    macd_fast = macd_fast_slider.get()
    macd_slow = macd_slow_slider.get()
    macd_signal = macd_signal_slider.get()
    rsi_periods = rsi_periods_slider.get()
    rsi_upper = rsi_upper_slider.get()
    rsi_lower = rsi_lower_slider.get()

    ta_dashboard(file, indicator, bb_k, bb_n, macd_fast, macd_slow, macd_signal, rsi_periods, rsi_upper, rsi_lower)

root = tk.Tk()
root.title("Technical Analysis Dashboard")

ttk.Label(root, text="Select File:").grid(column=0, row=0)
file_combobox = ttk.Combobox(root, values=files)
file_combobox.grid(column=1, row=0)

ttk.Label(root, text="Select Indicators:").grid(column=0, row=1)
indicator_listbox = tk.Listbox(root, selectmode=tk.MULTIPLE)
for ind in indicators:
    indicator_listbox.insert(tk.END, ind)
indicator_listbox.grid(column=1, row=1)

ttk.Label(root, text="Bollinger Bands k:").grid(column=0, row=2)
bb_k_slider = tk.Scale(root, from_=0.5, to=4.0, resolution=0.5,
                        orient=tk.HORIZONTAL)
bb_k_slider.grid(column=1, row=2)

ttk.Label(root, text="Bollinger Bands n:").grid(column=0, row=3)
bb_n_slider = tk.Scale(root, from_=1, to=40,
                        orient=tk.HORIZONTAL)
bb_n_slider.grid(column=1, row=3)

ttk.Label(root, text="MACD Fast Avg:").grid(column=0, row=4)
macd_fast_slider = tk.Scale(root, from_=2, to=50,
                             orient=tk.HORIZONTAL)
macd_fast_slider.grid(column=1, row=4)

ttk.Label(root, text="MACD Slow Avg:").grid(column=0,row=5)
macd_slow_slider = tk.Scale(root, from_=2,to=50,
                             orient=tk.HORIZONTAL)
macd_slow_slider.grid(column=1,row=5)

ttk.Label(root,text="MACD Signal:").grid(column=0,row=6)
macd_signal_slider = tk.Scale(root,
                               from_=2,to=50,
                               orient=tk.HORIZONTAL)
macd_signal_slider.grid(column=1,row=6)


ttk.Label(root,text="RSI Periods:").grid(column=0,row=7)
rsi_periods_slider = tk.Scale(root,
                               from_=2,to=50,
                               orient=tk.HORIZONTAL)
rsi_periods_slider.grid(column=1,row=7)

ttk.Label(root,text="RSI Upper Threshold:").grid(column=0,row=8)
rsi_upper_slider = tk.Scale(root,
                             from_=1,to=100,
                             orient=tk.HORIZONTAL)
rsi_upper_slider.grid(column=1,row=8)

ttk.Label(root,text="RSI Lower Threshold:").grid(column=0,row=9)
rsi_lower_slider = tk.Scale(root,
                             from_=1,to=100,
                             orient=tk.HORIZONTAL)
rsi_lower_slider.grid(column=1,row=9)


submit_button = ttk.Button(root,text="Submit", command=on_submit)
submit_button.grid(columnspan=2,row=10,pady=(10))

root.mainloop()