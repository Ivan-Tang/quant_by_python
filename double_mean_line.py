import tushare as ts
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

pro = ts.pro_api('b2aae2e87d25b396821ef5b49f762e916d68f9fa14f5262fc7b42710')


fund = 100000

hold = 0

tax_rate = 0

ts_code = '600519.SH'
start_date = '20200101'
end_date = '20240101'

df = pro.daily(ts_code= ts_code, start_date= start_date, end_date= end_date)

df['mean'] = (df['high'] + df['low'])/2

for i in range(20,len(df)-20):
    for j in range(i-4,i):
        fived_high_avg = np.sum(df['high'][j:j+5])/5
        fived_low_avg = np.sum(df['low'][j:j+5])/5
        mid_5 = (fived_high_avg + fived_low_avg)/2
    df.loc[i, '5d_high_avg'] = fived_high_avg
    df.loc[i, '5d_low_avg'] = fived_low_avg
    df.loc[i, '5d_mid'] = mid_5
    for j in range(i-20,i):
        twentyd_high_avg = np.sum(df['high'][j:j+20])/20
        twentyd_low_avg = np.sum(df['low'][j:j+20])/20
        mid_20 = (twentyd_high_avg + twentyd_low_avg)/2
    df.loc[i, '20d_high_avg'] = twentyd_high_avg
    df.loc[i, '20d_low_avg'] = twentyd_low_avg
    df.loc[i, '20d_mid'] = mid_20

mid_diff = df['5d_mid'] - df['20d_mid']
df['mid_diff'] = mid_diff



df['5d_ema'] = df['close'].ewm(span=5, adjust=False).mean()
df['20d_ema'] = df['close'].ewm(span=20, adjust=False).mean()
ema_diff = df['5d_ema'] - df['20d_ema']
df['ema_diff'] = ema_diff

    


df['buy_point'] = np.nan
df['sell_point'] = np.nan


for i in range(20, len(df)-20):
    if df.loc[i-1, 'ema_diff'] <= 0 and df.loc[i+1, 'ema_diff'] >= 0:
        df.loc[i, 'buy_point'] = 1
    if df.loc[i-1, 'ema_diff'] >= 0 and df.loc[i+1, 'ema_diff'] <= 0:
        df.loc[i,'sell_point'] = 1
        
# buy and sell strategy
for i in range(len(df)):
    if df.loc[i, 'buy_point'] == 1 and fund > 100*df.loc[i, 'open']:
        buy_price = df.loc[i, 'open']
        fund -= buy_price*100*(1+tax_rate)
        hold += 100
        print('buy at', buy_price, 'with fund', fund, 'hold', hold)
    if df.loc[i,'sell_point'] == 1 and hold >= 100:
        sell_price = df.loc[i, 'open']
        fund += sell_price*100*(1+tax_rate)
        hold -= 100
        print('sell at', sell_price, 'with fund', fund, 'hold', hold)
    df.loc[i, 'fund'] = fund + hold*df.loc[i, 'close']
    df.loc[i, 'hold'] = hold

print('final fund', fund)

# print the data
print(df)
plt.figure(figsize = (24,8))


plt.subplot(131)
plt.title('Stock Price and BS points')
plt.xlabel('Date')
plt.ylabel('Price(CNY)')
plt.plot(df['trade_date'].to_numpy(), df['mean'].to_numpy(), label='mean')
plt.plot(df['trade_date'].to_numpy(), df['5d_mid'].to_numpy(), label='5d_mid')
plt.plot(df['trade_date'].to_numpy(), df['20d_mid'].to_numpy(), label='20d_mid')
plt.plot(df['trade_date'].to_numpy(), df['5d_ema'].to_numpy(), label='5d_ema')
plt.plot(df['trade_date'].to_numpy(), df['20d_ema'].to_numpy(), label='20d_ema')

for i in range(len(df)):
    if df.loc[i, 'buy_point'] == 1:
        plt.scatter(df.loc[i, 'trade_date'], df.loc[i, 'close'], color='green')
    if df.loc[i,'sell_point'] == 1:
        plt.scatter(df.loc[i, 'trade_date'], df.loc[i, 'close'], color='red')

plt.legend()

plt.subplot(132)
plt.title('Fund')
plt.xlabel('Date')
plt.ylabel('Fund(CNY)')
plt.plot(df['trade_date'].to_numpy(), df['fund'].to_numpy(), label='fund')
plt.scatter(df.loc[df['sell_point'] == 1, 'trade_date'], df.loc[df['sell_point'] == 1, 'fund'], color='green')



plt.subplot(133)
plt.title('Hold')
plt.xlabel('Date')
plt.ylabel('Hold(Shares)')
plt.plot(df['trade_date'].to_numpy(), df['hold'].to_numpy(), label='hold')

plt.tight_layout()
plt.show()




        

