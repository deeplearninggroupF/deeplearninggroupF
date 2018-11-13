import pandas as pd
data = pd.read_csv('target.csv',index_col = 0)
data = data.sort_values(['Date'],ascending = False)
data = data.reset_index(drop = True)
date = list(data.Date.unique())

def normalize(a,mean,std):
    return (a-mean)/std
    

for i in range(13):
    study_date = date[250*i : 250*(i+4)]        
    study_period = data[data.Date.isin(study_date)]
    
    trade_date = date[250*i : 250*(i+1)]
    trade_data = data[data.Date.isin(trade_date)]
    
    train_date = date[250*(i+1) : 250*(i+4)]
    train_data = data[data.Date.isin(train_date)]
    
    ticker_list = trade_data.Ticker.unique()
    output = pd.DataFrame()
    for ticker in ticker_list:
        train_mean = trade_data[data.Ticker ==ticker].Return.mean()
        train_std = trade_data[data.Ticker == ticker].Return.std()
        study_period_new = study_period[study_period.Ticker == ticker]
        study_period_new['Normalized_Return'] = study_period_new['Return'].apply(normalize,args=(train_mean,train_std))
        output = pd.concat([output,study_period_new], ignore_index = True)
    break
    output.to_csv('normalized_return_'+str(i)+'.csv')
    print('finish')
        
        
    
    
    