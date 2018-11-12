import pandas as pd
import datetime 

f = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')
data = pd.read_csv('data.csv',header = 0,index_col=0,converters = {'Date':f})

Ticker_List = data.Ticker.unique()
output = pd.DataFrame()

for ticker in Ticker_List:
    sub_data  = data[data.Ticker == ticker]
    sub_data = sub_data.sort_values(['Date'])
    sub_data = sub_data.reset_index()
    date_list = sub_data['Date']
    date_diff = date_list.diff()
    if (date_diff > datetime.timedelta(days =10)).any():
        split_index = list(date_diff.index[date_diff > datetime.timedelta(days = 10)])
        split_index += [0,len(sub_data.index)]
        return_data = pd.DataFrame()
        for i in range(len(split_index)-1):
            sub_sub_data = sub_data.iloc[split_index[i]:split_index[i+1]]
            sub_sub_data['Return'] = sub_sub_data.Price.pct_change()
            return_data = pd.concat([return_data, sub_sub_data],ignore_index = True)
            return_data = return_data.drop(['index'], axis=1)
    else:
        sub_data['Return'] = sub_data.Price.pct_change()
        sub_data = sub_data.drop(['index'],axis = 1)
        return_data = sub_data

    output = pd.concat([output,return_data],ignore_index = True)    
output.to_csv('return.csv')
    
    