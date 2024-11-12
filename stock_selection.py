import tushare as ts 
import os
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import mplfinance as mpf
import akshare as ak
import numpy as np
from tqdm import tqdm


# set working directory
os.chdir('/Users/IvanTang/quant')

# get data from tushare
pro = ts.pro_api('b2aae2e87d25b396821ef5b49f762e916d68f9fa14f5262fc7b42710') 

basic = pd.read_csv('stock_basic.csv')

industry = '元器件'
start_date = '20240101'
end_date = '20241031'

# get industry list

industries = basic['industry'].unique()

preselected_stocks= basic.loc[basic['industry'].isin([industry]), 'ts_code']


# set stock codes   

codes = preselected_stocks.str[:6].tolist()
ts_codes = basic.loc[basic['ts_code'].str[:6].isin(codes)]['ts_code'].tolist()
print('当前板块为:',industry,'，共',len(ts_codes),'只股票')



def clean_data(daily, moneyflow):
    #清洗数据
    daily = pd.merge(daily, moneyflow, on='trade_date', how='inner')
    daily['trade_date'] = pd.to_datetime(daily['trade_date'])
    daily = daily.reset_index(drop=True)
    daily.sort_values(by = 'trade_date', inplace=True)
    daily.rename(columns={'trade_date': 'date'}, inplace=True)
    daily.rename(columns={'vol': 'volume'}, inplace=True)
    daily.set_index('date', inplace=True)

    daily['5d_diff'] = daily['close'].shift(-5) - daily['open']
    daily['net_mf_amount_shifted1']  = daily['net_mf_amount'].shift(1)
    
    daily = daily.dropna(subset=['close', 'open', 'high', 'low', 'volume', 'amount', 'net_mf_vol', 'net_mf_amount', 'net_mf_amount_shifted1', '5d_diff'])
    return daily

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def ma(series, window):
    return series.rolling(window=window).mean()

#计算HHJSJD
def get_HHJSJD(daily):
    daily['HHJSJDA'] = (3 * daily['close'] + daily['open'] + daily['low'] + daily['high']) / 6

    # 创建一个空的列表来存储加权和
    weights = []
    for i in range(20):
        weight = (20 - i) * daily['HHJSJDA'].shift(i)
        weights.append(weight)

    # 将所有权重相加并除以 210
    daily['HHJSJDB'] = sum(weights) / 210
    daily['HHJSJDC'] = daily['HHJSJDB'].rolling(window=5).mean()

    daily['HHJSJDA_shifted1'] = daily['HHJSJDA'].shift(1)
    daily['HHJSJDB_shifted1'] = daily['HHJSJDB'].shift(1)
    daily['HHJSJDC_shifted1'] = daily['HHJSJDC'].shift(1)
    daily['HHJSJD_diff'] = daily['HHJSJDB'] - daily['HHJSJDC']
    daily['HHJSJD_diff2'] = daily['HHJSJD_diff'] - daily['HHJSJD_diff'].shift(1)

    daily = daily.dropna(subset=['HHJSJDA','HHJSJDB','HHJSJDC'])

    return daily

#计算XYS
def get_XYS(daily,capital):

    # 计算WY1001
    daily['WY1001'] = (2 * daily['close'] + daily['high'] + daily['low']) / 3

    # 计算WY1002, WY1003, WY1004
    daily['WY1002'] = ema(daily['WY1001'], 3)
    daily['WY1003'] = ema(daily['WY1002'], 3)
    daily['WY1004'] = ema(daily['WY1003'], 3)

    # 计算XYS0
    daily['XYS0'] = (daily['WY1004'] - daily['WY1004'].shift(1)) / daily['WY1004'].shift(1) * 100

    # 计算PJGJ
    daily['PJGJ'] = daily['amount'] / daily['volume'] / 100

    # 计算SSRYDJX, SSRCJL, SSRCJE, SSRCBJX
    daily['SSRYDJX'] = ma(daily['PJGJ'], 13)
    daily['SSRCJL'] = ema(daily['volume'], 13)
    daily['SSRCJE'] = ema(daily['amount'], 13)
    daily['SSRCBJX'] = daily['SSRCJE'] / daily['SSRCJL'] / 100

    # 计算CYS13
    daily['CYS13'] = (daily['close'] - daily['SSRCBJX']) / daily['SSRCBJX'] * 100

    # 计算XYSHSL
    daily['XYSHSL'] = ema(daily['volume'] / capital * 100, 13)

    # 计算XYS1和XYS2
    daily['XYS1'] = ma(daily['XYS0'], 1)
    daily['XYS2'] = ma(daily['XYS0'], 2)
    daily['XYS1_shifted1'] = daily['XYS1'].shift(1)
    daily['XYS2_shifted1'] = daily['XYS2'].shift(1)
    daily['XYS_diff'] = daily['XYS2'] - daily['XYS1']
    daily['XYS_diff2'] = daily['XYS_diff'] - daily['XYS_diff'].shift(1)

    # 清洗数据 去除空值
    daily = daily.dropna(subset=['WY1001', 'WY1002', 'WY1003', 'WY1004', 'XYS0', 'PJGJ', 'SSRYDJX', 'SSRCJL', 'SSRCJE', 'SSRCBJX', 'CYS13', 'XYSHSL', 'XYS1', 'XYS2'])
    return daily

def HHJSJD_gold_cross(daily,indice,threshold):
    #判断金叉
    threshold = threshold
    if(daily['HHJSJD_diff2'].iloc[-indice] > threshold and daily['HHJSJD_diff'].iloc[-indice-1] < 0 < daily['HHJSJD_diff'].iloc[-indice]):
        return True
    else:
        return False
    
def XYS_gold_cross(daily,indice,threshold):
    #判断金叉
    threshold = threshold
    if(daily['XYS_diff2'].iloc[-indice] > threshold and daily['XYS_diff'].iloc[-indice-1] < 0 < daily['XYS_diff'].iloc[-indice]):
        return True
    else:
        return False
    
#判断死叉
def HHJSJD_death_cross(daily,indice,threshold):

    if(daily['HHJSJD_diff'].iloc[-indice] < -threshold and daily['HHJSJD_diff'].iloc[-indice-1] > 0 > daily['HHJSJD_diff'].iloc[-indice]):
        return True
    else:
        return False
    
def XYS_death_cross(daily,indice,threshold):
    if(daily['XYS_diff2'].iloc[-indice] < -threshold and daily['XYS_diff'].iloc[-indice-1] > 0 > daily['XYS_diff'].iloc[-indice]):
        return True
    else:
        return False

def positive_vol(moneyflow,indice,min_vol):
    if(all(moneyflow['net_mf_vol'].iloc[i] > min_vol for i in range(-indice,-1)) ):
        return True

def negative_vol(moneyflow,indice,min_vol):
    if(all(moneyflow['net_mf_vol'].iloc[i] < -min_vol for i in range(-indice,-1)) ):
        return True


def get_daily(start_date, end_date, ts_code):
    #处理daily,moneyflow数据，计算各项指标
    start_date = str(start_date)
    end_date = str(end_date)
    daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    moneyflow = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    # get data from akshare
    stock_individual_info_em_df = ak.stock_individual_info_em(symbol=ts_code[:6])
    capital_value = stock_individual_info_em_df.loc[stock_individual_info_em_df['item'] == '流通股', 'value']
    capital = float(capital_value.values[0])
    daily = clean_data(daily, moneyflow)
    daily = get_alphas(daily)
    daily =get_HHJSJD(daily)
    daily = get_XYS(daily, capital)

    return daily

def select_stock(threshold,start_date,end_date):
    start_date = str(start_date)
    end_date = str(end_date)
    threshold = threshold
    selected_stocks = []
    for i in tqdm(range(len(ts_codes)-1)):
        code = codes[i]
        ts_code = ts_codes[i]
        # get data from tushare
        daily = get_daily(start_date, end_date, ts_code)
        try:
            # 筛选股票
            if (any(XYS_gold_cross(daily, i, threshold) for i in range(3))):
                selected_stocks.append(ts_code)
        except IndexError as e:
            print(f"IndexError: {e}, 股票代码为 {ts_code}， daily 的行数为 {len(daily)}，无法进行索引。")

        
    print(f"选股数量：{len(selected_stocks)}",'代码为：',selected_stocks)

def plot_stock(start_date,end_date,ts_code,):
    # get data 
    daily = get_daily (start_date, end_date, ts_code)
    #画图
    apd = [
        mpf.make_addplot(daily['net_mf_amount'].values, color='orange', panel=1, ylabel='net_mf_amount'),
        mpf.make_addplot(daily['HHJSJDA'].values, color='blue', panel=0),
        mpf.make_addplot(daily['HHJSJDB'].values, color='yellow', panel=0),
        mpf.make_addplot(daily['HHJSJDC'].values, color='red', panel=0),
        mpf.make_addplot(daily['XYS1'].values, color='yellow', panel=1),
        mpf.make_addplot(daily['XYS2'].values, color='purple', panel=1)

    ]

    mpf.plot(daily, type='candle', volume=True, addplot=apd, style='charles', title=ts_code,)

def get_alphas(df):
    # 计算过去9天最低价的排名，然后取相反数
    alpha_3 = -df['low'].rank(pct=True).rolling(window=9).min().shift(1)

    # 计算过去10天开盘价和成交量之间的相关性，然后取相反数
    # 使用 DataFrame 的滚动窗口计算相关性
    rolling_corr = df['open'].rolling(window=10).corr(df['volume'])

    # 取负值以得到 alpha_6
    alpha_6 = -rolling_corr


    # 检查过去5天内收盘价变化的最小值和最大值，并根据这些值的正负来确定Alpha因子的值
    delta_close = df['close'].diff()  # 计算收盘价的日变化量
    ts_min_delta_close = delta_close.rolling(window=5).min()  # 过去5天的最小变化量
    ts_max_delta_close = delta_close.rolling(window=5).max()  # 过去5天的最大变化量
    alpha_9 = np.where((ts_min_delta_close > 0) | (ts_max_delta_close < 0), delta_close, -delta_close)

    # 计算收盘价的时间序列排名等
    ts_rank_close = df['close'].rank(pct=True).rolling(window=10).apply(lambda x: x.iloc[-1], raw=False)
    delta_delta_close = delta_close.diff()  # 收盘价变化的二阶变化
    ts_rank_delta_delta_close = delta_delta_close.rank(pct=True).rolling(window=1).apply(lambda x: x.iloc[-1], raw=False)
    adv20 = df['volume'].rolling(window=20).mean()  # 过去20天平均成交量
    ts_rank_volume_adv20 = (df['volume'] / adv20).rank(pct=True).rolling(window=5).apply(lambda x: x.iloc[-1], raw=False )
    alpha_17 = (-1 * ts_rank_close) * ts_rank_delta_delta_close * ts_rank_volume_adv20

    # 计算收益率的相反数、过去20天平均成交量和成交量加权平均价的乘积
    returns = df['close'].pct_change()  # 计算收益率
    vwap = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()  # 成交量加权平均价
    alpha_25 = (-returns * adv20 * vwap).rank(pct=True) * (df['high'] - df['close'])

    # 将计算出的Alpha因子添加到原始DataFrame中
    df['alpha_3'] = alpha_3
    df['alpha_6'] = alpha_6
    df['alpha_9'] = alpha_9
    df['alpha_17'] = alpha_17
    df['alpha_25'] = alpha_25

     # 进行位移处理
    df['alpha_3_shifted'] = df['alpha_3'].shift(1)
    df['alpha_6_shifted'] = df['alpha_6'].shift(1)
    df['alpha_9_shifted'] = df['alpha_9'].shift(1)
    df['alpha_17_shifted'] = df['alpha_17'].shift(1)
    df['alpha_25_shifted'] = df['alpha_25'].shift(1)

    # 清洗数据，删除因位移而产生的 NaN 值
    df.dropna(subset=['alpha_3_shifted', 'alpha_6_shifted', 
                     'alpha_9_shifted', 'alpha_17_shifted', 
                     'alpha_25_shifted'], inplace=True)

    return df


# df = pd.read_csv('data.csv')  # 假设这个文件里有你需要的数据
# result_df = get_alphas(df)
# print(result_df[['alpha_3', 'alpha_6', 'alpha_9', 'alpha_17', 'alpha_25']])

data = get_daily('20160101','20211231','600519.SH')
data.to_csv('data.csv')
new_data = get_daily('20210101','20241031','600519.SH')
new_data.to_csv('new_data.csv')


'''施工中'''
def merge_hold(hold):
    # 合并持仓股票
    if len(hold) > 1:
        for i in range(len(hold)):
            for j in range(i+1, len(hold)):
                if hold[i][0] == hold[j][0]:
                    hold[i][1] += hold[j][1]
                    hold[i][2] = (hold[i][1] * hold[i][2] + hold[j][1] * hold[j][2]) / (hold[i][1] + hold[j][1])
                    hold.pop(j)
                    break
                return hold
    else:
        return hold
    

def backtest(df, start_date, end_date):
    fund = 1000000
    hold = []
    tax_rate = 0.0001

    start_date = str(start_date)
    end_date = str(end_date)
    # 安排日期
    days = pd.date_range(start=start_date, end=end_date)
    weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-FRI')

    for week_start in weekly_dates:
        # 每周选择股票
        selected_stocks = select_stock(0.3, week_start.strftime('%Y%m%d'), str((week_start)+pd.offsets.Week(1)).strftime('%Y%m%d'))
        
        if not selected_stocks:
            continue  # 没有股票可选择

        xys_diff2_values = []

        for stock in selected_stocks:
            #获取股票的日线数据和指标
            daily = get_daily((week_start - pd.offsets.Day(20)).strftime('%Y%m%d'), str((week_start) + pd.offsets.Week(weekday=4)).strftime('%Y%m%d'), stock)
            # 计算XYS_diff2最大的股票
            xys_diff2_values.append([np.mean(daily.loc[week_start:week_start + pd.offsets.Week(weekday=4), stock]['XYS_diff2'])])
        max_stock_index = np.argmax(xys_diff2_values)
        max_stock = selected_stocks[max_stock_index]

        # 计算可以投资的金额
        max_investment = fund * 0.1  # 投资10%的资金
        df = get_daily(week_start.strftime('%Y%m%d'), str((week_start) + pd.offsets.Week(weekday=4)).strftime('%Y%m%d'), max_stock)
        buy_price = df.loc[week_start + pd.offsets.Week(weekday=4)]['close']  # 计算买入的价格
        buy_quantity = max_investment // df.loc[week_start + pd.offsets.Week(weekday=4)]['close']  # 计算买入的股票数量

        # 买入
        if buy_quantity > 0:
            fund -= buy_quantity * buy_price * (1 + tax_rate)  # 扣除费用
            hold.append([max_stock, buy_quantity, buy_price])  # 记录持仓股票，持仓数量，持仓价格
            hold = merge_hold(hold)  # 合并持仓股票
            print(f"在 {week_start.date()} 买入 {buy_quantity} 股 {hold}，当前资金: {fund:.2f}")
        
    for day in days:
        # 检查持仓股票是否出现死叉
        for stock in hold:
            code = stock[0]
            df = get_daily(str((day) - pd.offsets.Day(30)).strftime('%Y%m%d'), day.strftime('%Y%m%d'),code)
            # 如果出现死叉，则卖出
            if XYS_death_cross(df, 1, 0.3):
                fund += position * df.loc[day] * (1 - tax_rate)  # 返回资金
                print(f"在 {day.date()} 卖出 {position} 股 {hold}，当前资金: {fund:.2f}")
                position = 0
                hold = None
            else:
                # 否则，加仓10%
                additional_investment = fund * 0.1  # 额外投资10%的资金
                additional_quantity = additional_investment // df.loc[next_week_start][hold]  # 计算加仓数量

                if additional_quantity > 0:
                    fund -= additional_quantity * df.loc[next_week_start][hold] * (1 + transaction_fee)  # 扣除费用
                    position += additional_quantity
                    print(f"在 {next_week_start.date()} 加仓 {additional_quantity} 股 {hold}，当前资金: {fund:.2f}")

    # 输出最终资金
    if position > 0:
        fund += position * df.loc[next_week_start][hold] * (1 - tax_rate)  # 卖出剩余股票
        print(f"最终资金: {fund:.2f} (持有 {position} 股 {hold}，最后价格: {df.loc[next_week_start][hold]:.2f})")
    else:
        print(f"最终资金: {fund:.2f}")

    








