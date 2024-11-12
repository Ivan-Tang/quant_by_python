import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 初始化 Tushare API
pro = ts.pro_api('b2aae2e87d25b396821ef5b49f762e916d68f9fa14f5262fc7b42710')  # 替换为你的 Tushare token

# 获取股票数据以及数据清洗
ts_code = '600536.SH'  # 示例股票
fund = 2000000
funds_history = []
hold = 0
holds_history = []

tax_rate = 0.0001
start_date = '20200101'
end_date = '20240101'

df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df.set_index('trade_date', inplace=True)
df.sort_values(by='trade_date', ascending=True, inplace=True)

# 计算 MACD 指标
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_short'] = data['close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

df = compute_macd(df)

# 检测背离
def detect_divergence(data):
    buy_signals = []
    for i in range(1, len(data)):
        # 检查 MACD 背离，使用 iloc 获取位置索引
        if data['close'].iloc[i] > data['close'].iloc[i-1] and data['MACD'].iloc[i] < data['MACD'].iloc[i-1]:
            buy_signals.append((i, data['close'].iloc[i]))  # 记录背离信号，存储索引及收盘价
    return buy_signals

buy_signals = detect_divergence(df)

# 检测卖出信号
def detect_sell_signals(data):
    sell_signals = []
    for i in range(1, len(data)):
        if data['close'].iloc[i] < data['close'].iloc[i-1] and data['MACD'].iloc[i] > data['MACD'].iloc[i-1]:
            sell_signals.append((i, data['close'].iloc[i]))  # 存储索引位置
    return sell_signals

sell_signals = detect_sell_signals(df)

# 输出背离信号
if buy_signals:
    print("发现买入信号：")
    for signal in buy_signals:
        print(f"日期: {df.index[signal[0]].date()}, 股票价格: {signal[1]}")
else:
    print("未发现买入信号。")

# 输出卖出信号
if sell_signals:
    print("发现卖出信号：")
    for signal in sell_signals:
        print(f"日期: {df.index[signal[0]].date()}, 股票价格: {signal[1]}")
else:
    print("未发现卖出信号。")



#计算交易盈亏
for i in range(len(df)):
    if i in [s[0] for s in buy_signals] and fund >= 100*df.loc[df.index[i], 'open']:
        hold += 100
        fund -= 100*df.loc[df.index[i], 'open']*(1+tax_rate)
        print(f"于日期: {df.index[i].date()}, 买入: {100}")

    elif i in [s[0] for s in sell_signals] and hold >= 200:
        hold -= 100
        fund += 100*df.loc[df.index[i], 'open']*(1-tax_rate)
        print(f"于日期: {df.index[i].date()}, 卖出: {100}")
    funds_history.append(fund+hold*df.loc[df.index[i], 'close'])
    holds_history.append(hold)


print(f"最终资金: {fund+hold*df['close'].iloc[-1]}")






# 可视化股票收盘价和 MACD
plt.figure(figsize=(14, 10))

# 绘制收盘价
plt.subplot(221)
plt.plot(df.index.to_numpy(), df['close'].to_numpy(), label='Close Price', color='blue')
plt.title('Close Price')

buy_index =[]
sell_index = []

for signal in buy_signals: 
    buy_index.append(signal[0])
for signal in sell_signals:
    sell_index.append(signal[0])
plt.scatter(df.index[buy_index], df['close'].iloc[buy_index], marker='o', color='green', label='Buy Signal')
plt.scatter(df.index[sell_index], df['close'].iloc[sell_index], marker='x', color='red', label='Sell Signal')

plt.legend()



# 绘制 MACD
plt.subplot(222)
plt.plot(df.index.to_numpy(), df['MACD'].to_numpy(), label='MACD', color='blue')
plt.plot(df.index.to_numpy(), df['Signal'].to_numpy(), label='Signal Line', color='orange')
plt.title('MACD')
plt.legend()

# 绘制资金曲线
plt.subplot(223)
plt.plot(df.index.to_numpy(), funds_history, label='Fund', color='blue')
plt.scatter(df.index[len(df)-1], funds_history[len(df)-1], marker='o', color='green', label='Final Fund')
plt.title('Fund')

plt.subplot(224)
plt.plot(df.index.to_numpy(), holds_history, label='Hold', color='blue')
plt.title('Hold')

plt.tight_layout()
plt.show()
