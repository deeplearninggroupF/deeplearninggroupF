import pandas as pd
data= pd.read_csv('return.csv',index_col = 0)
data = data[['Date', 'Ticker', 'Return']]
data = data.dropna()
data = data.sort_values(['Date','Ticker'])
data = data.reset_index(drop = True)
date_list = data.Date.unique()
output = pd.DataFrame()

def target_class(equity_return, median):
    if equity_return >= median:
        return 1
    else:
        return 0
    
for date in date_list:
    sub_data = data[data.Date == date]
    return_median = sub_data['Return'].median()
    sub_data['Target'] = sub_data['Return'].apply(target_class,args=(return_median,))
    output = pd.concat([output,sub_data], ignore_index = True)

output.to_csv('target.csv')

    
    



