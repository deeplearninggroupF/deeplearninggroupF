"""
In this portfolio, we long top k stocks with the highest possibility to beat cross-sectional median
and short top K stocks with the lowest possibility to beat cross-sectional median.
All stocks equal weighted and prior to transaction costs.

"""

def prediction(a):
    if a < 0.5:
        return 0
    else:
        return 1
    
import pandas as pd

# Generate Portfolio Holdings
long_portfolio = pd.DataFrame()
short_portfolio = pd.DataFrame()
for i in range(13):
    result = pd.read_csv('C:/Users/Yu/Desktop/Pred/Pred/prediction_period_'+str(i)+'.csv',index_col = 0)
    result['y_hat'] = result['pred'].apply(prediction)
    result.columns = ['y_prob','y_true','Ticker','Date','y_hat']
    result = result[['y_hat','y_true','y_prob','Date','Ticker']]
    result = result.reset_index(drop = True)
    return_info = pd.read_csv('C:/Users/Yu/Desktop/Normalize_Return/normalized_return_'+str(i)+'.csv',index_col = 0)
    date_list = result.Date.unique()
    long_p = pd.DataFrame()
    short_p = pd.DataFrame()
    for date in date_list:
        sub_result = result[result.Date == date]
        sub_result = sub_result.sort_values(['y_prob'],ascending = True)
        short = sub_result.iloc[:10]
        long = sub_result.tail(10)
        short_p = pd.concat([short_p,short],ignore_index = True)
        long_p = pd.concat([long_p,long], ignore_index = True)
    
    long_hold = long_p.merge(return_info,how = 'inner',on =['Date','Ticker'],suffixes = ('_x', '_y'))
    short_hold = short_p.merge(return_info,how = 'inner',on =['Date','Ticker'],suffixes = ('_x', '_y'))
    
    long_hold = long_hold[['Date','Ticker','Return']]    
    short_hold = short_hold[['Date','Ticker','Return']]
    
    long_portfolio = pd.concat([long_portfolio,long_hold],ignore_index = True)
    short_portfolio = pd.concat([short_portfolio,short_hold],ignore_index = True)

long_date = len(list(long_portfolio.Date.unique()))
short_date = len(list(short_portfolio.Date.unique()))

long_return = long_portfolio['Return'].sum()/long_date/10
short_return = (-1)*short_portfolio['Return'].sum()/short_date/10

print(long_return)
print(short_return)
print((long_return+short_return)/2)