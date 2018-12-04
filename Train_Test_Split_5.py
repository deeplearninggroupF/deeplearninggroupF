import pandas as pd

for m in range(13):
    data = pd.read_csv('normalized_return_'+str(m)+'.csv',index_col = 0)
    data = data[['Date','Ticker','Normalized_Return','Target']]
    data = data.sort_values(['Date','Ticker'],ascending = True)
    date_list = list(data.Date.unique())
    ticker_list = list(data.Ticker.unique())

    # Generate Training Set
    final_output = pd.DataFrame()

    for ticker in ticker_list:
        output = pd.DataFrame()
        for i in range(510):
            sub_date_list = date_list[i : (i+241)]
            ticker_data = data[data.Ticker == ticker]
            ticker_date_data = ticker_data[ticker_data.Date.isin(sub_date_list)]
            ticker_date_data = ticker_date_data.reset_index(drop = True)
            if len(ticker_date_data.index) >= 241:
                ticker_date_data = ticker_date_data.transpose()
                name = ticker_date_data.iloc[1,0]
                target = ticker_date_data.iloc[-1,-1]
                target_date = ticker_date_data.iloc[0,-1]
                ticker_date_data = ticker_date_data.drop(['Date','Ticker','Target'],axis = 0)
                ticker_date_data = ticker_date_data.iloc[:,0:-1]
                ticker_date_data['target'] = target
                ticker_date_data['ticker'] = name
                ticker_date_data['target_date'] = target_date
                ticker_date_data = ticker_date_data.reset_index(drop= True)
                output = pd.concat([output,ticker_date_data],ignore_index = True)
        print(str(ticker))
        final_output = pd.concat([final_output,output],ignore_index = True)

    final_output.to_csv('Set_'+str(m)+'_Train.csv')
    
    # Generate Testing Set
    final_output_2 = pd.DataFrame()
    
    for ticker in ticker_list:
        output_2 = pd.DataFrame()
        for i in range(510,760):
            sub_date_list = date_list[i:(i+241)]
            ticker_data = data[data.Ticker == ticker]
            ticker_date_data = ticker_data[ticker_data.Date.isin(sub_date_list)]
            ticker_date_data = ticker_date_data.reset_index(drop = True)
            if len(ticker_date_data.index) >= 241:
                ticker_date_data = ticker_date_data.transpose()
                name = ticker_date_data.iloc[1,0]
                target = ticker_date_data.iloc[-1,-1]
                target_date = ticker_date_data.iloc[0,-1]
                ticker_date_data = ticker_date_data.drop(['Date','Ticker','Target'],axis = 0)
                ticker_date_data = ticker_date_data.iloc[:,0:-1]
                ticker_date_data['target'] = target
                ticker_date_data['ticker'] = name
                ticker_date_data['target_date'] = target_date
                ticker_date_data = ticker_date_data.reset_index(drop= True)
                output_2 = pd.concat([output_2,ticker_date_data],ignore_index = True)
        print(str(ticker))
        final_output_2 = pd.concat([final_output_2,output_2],ignore_index = True)

    final_output_2.to_csv('Set_'+str(m)+'_Test.csv')
