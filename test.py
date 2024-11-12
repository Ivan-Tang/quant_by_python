import akshare as ak

def get_index_stocks(index_code):
    # 获取上证指数成分股
    hs300_stocks_df =ak.index_stock_cons(symbol=index_code)
    hs300_stocks = hs300_stocks_df['品种代码'].tolist()
    return hs300_stocks

df = ak.stock_individual_info_em(symbol="000001")
print(df)
index_code = "000300.SH"
#hs300_stocks = get_index_stocks(index_code)

hs300_stocks_df =ak.index_stock_cons(symbol=index_code)
print(hs300_stocks_df)
'''
for stock in hs300_stocks:
    industries = []
    df = ak.stock_individual_info_em(symbol="000001")
    industry_name = df.loc[df['item'] == '行业', 'value']
    industries.append(industry_name)

    print(stock, industries)

'''


