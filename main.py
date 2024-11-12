import akshare as ak
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from datetime import datetime, timedelta
import time

# 全局变量
days = 0
refresh_rate = 10
stocknum = 10
current_date = datetime.now().strftime("%Y%m%d")

trade_dates = ak.trade_calendar(start_date='20000101', end_date=current_date)
trade_dates = trade_dates['calendarDate'].tolist()

sw_index_first_info_df = ak.sw_index_first_info()
industry_matrix = sw_index_first_info_df.set_index('行业代码')['行业名称'].to_dict()

# 获取数据

def get_index_stocks(index_code):
    # 获取上证指数成分股
    hs300_stocks_df =ak.index_stock_cons(symbol=index_code)
    hs300_stocks = hs300_stocks_df['品种代码'].tolist()
    return hs300_stocks


def get_stock_info(stock, start_date, end_date):
    # 股票基本面数据
    # 在start_date和end_date之间且在trade_dates中的天，获取该股票数据，返回一个dataframe

    global trade_dates
    date_list = []
    for date in trade_dates:
        if start_date <= date <= end_date:
            date_list.append(date)
            if len(date_list) == 0:
                continue

    date_list.sort()
    df_list = []


    for date in date_list:
        #流通市值
        stock_individual_info_em_df = ak.stock_individual_info_em(symbol=stock)
        market_cap = stock_individual_info_em_df.loc[stock_individual_info_em_df['item'] == '流通市值']['value'].values[0]
        available_num = stock_individual_info_em_df.loc[stock_individual_info_em_df['item'] == '流通股']['value'].values[0]

        industry = stock_individual_info_em_df.loc[stock_individual_info_em_df['item'] == '行业']['value']


        #净资产
        stock_individual_spot_xq_df = ak.stock_individual_spot_xq(symbol=stock)
        net_capital = stock_individual_spot_xq_df.loc[stock_individual_spot_xq_df['item'] == '资产净值/总市值', 'value'].values[0]

        #负债率
        stock_zcfz_em_df = ak.stock_zcfz_em(date=date)
        lev = stock_zcfz_em_df.loc[stock_zcfz_em_df['股票代码'] == stock, '资产负债率'].values[0]

        # 净利润
        stock_financial_benefit_ths_df = ak.stock_financial_benefit_ths(symbol=stock, indicator="按报告期")
        net_profit = net_profit = stock_financial_benefit_ths_df.iloc[0]['*净利润']
        net_profit_value = float(net_profit.replace('亿', '')) * 1e8

        # 营业收入增长率
        stock_financial_analysis_indicator_df = ak.stock_financial_analysis_indicator(symbol=stock)
        g = stock_financial_analysis_indicator_df['主营业务收入增长率(%)'].iloc[-1] * 0.01

        df = pd.DataFrame({'date': [date],  '流通股数': [available_num], '行业' : [industry], '净资产_亿元': [net_capital/1e8], 
                        '净利润_亿元': [net_profit_value/1e8],  '资产负债率': [lev], '主营业务收入增长率(%)': [g],  '流通市值_亿元': [market_cap/1e8],})

        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

def get_fundamentals(stock_list):
    # stock_list为股票列表，获取每只股票的基本面数据，返回一个dataframe
    df_list = []
    for stock in stock_list:
        df = get_stock_info(stock)
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


def preprocess_data(df):
    # 对数变换
    df['log_MC'] = np.log(df['流通市值_亿元'])
    df['log_NC'] = np.log(df['净资产_亿元'])
    df['log_NP'] = np.log(df['净利润_亿元'])

    #行业编码
    df['行业'] = df['行业'].map(industry_matrix)
    df = pd.get_dummies(df, columns=['行业'])

    # 缺失值处理
    df.fillna(0, inplace=True)
    return df


def train_model(X, Y):
    # 训练模型, 输入为：行业变量，对数净资产，对数净利润，资产负债率，主营业务收入增长率；输出为对数流通市值
    svr = SVR(kernel='rbf', gamma=0.1)
    svr.fit(X, Y)
    return svr

def predict_stock_price(stock_code, svr, start_date, end_date):
    # 获取股票数据
    stock_df = get_stock_info(stock_code, start_date, end_date) # 获取股票数据
    # 预处理数据
    stock_df = preprocess_data(stock_df)
    # 预测股价
    X = stock_df.iloc[:, 2:-1].values
    Y = svr.predict(X)
    return Y[-1]

def main():
    # 获取指数成分股
    index_code = "000300.SH"
    hs300_stocks = get_index_stocks(index_code)

    start_date = '20150101'
    end_date = '20211231'
    # 获取基本面数据
    fundamental_df = get_fundamentals(hs300_stocks, start_date, end_date)
    # 预处理数据
    X = fundamental_df.iloc[:, 2:-1].values
    Y = fundamental_df.iloc[:, -1].values
    X = preprocess_data(X)
    # 训练模型
    svr = train_model(X, Y)

    # 对stock_list中每只股票进行预测
    stock_list = hs300_stocks
    error_list = []
    for stock in stock_list:
        # 预测流通市值
        predict_log_MC = predict_stock_price(stock, svr, days)
        # 获取流通市值
        predict_MC = np.exp(predict_log_MC)
        true_MC = fundamental_df.loc[fundamental_df['股票代码'] == stock, '流通市值_亿元'].values[0]
        # 计算误差
        error = (predict_MC - true_MC) / true_MC * 100
        # 记录误差，附加股票代码
        error_list.append([stock, error])
        print(f"{stock} 误差：{error:.2f}%")

    # 输出误差排序
    error_list.sort(key=lambda x: x[1], reverse=True)
    



if __name__ == '__main__':
    main()