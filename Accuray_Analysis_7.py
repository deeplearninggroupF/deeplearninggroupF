# Analyze accuracy and generate long-short portfolio holdings
def prediction(a):
    if a < 0.5:
        return 0
    else:
        return 1

import pandas as pd
accu = 0
for i in range(13):
    result = pd.read_csv('C:/Users/Yu/Desktop/Pred/Pred/prediction_period_'+str(i)+'.csv',index_col = 0)
    result['y_hat'] = result['pred'].apply(prediction)
    result.columns = ['y_prob','y_true','Ticker','Date','y_hat']
    result = result[['y_hat','y_true','y_prob','Date','Ticker']]
    result = result.reset_index(drop = True)
    date_list = result.Date.unique()
    # K represents how many stocks we select to long or short for each trading day
    K = [10]
    for k in K:
        accurate_item = 0
        for date in date_list:
            sub_result = result[result.Date == date]
            sub_result = sub_result.sort_values(['y_prob'],ascending =True)
            short_df = sub_result.iloc[:k]
            long_df = sub_result.tail(k)
            short_long_df = pd.concat([short_df,long_df],ignore_index = True)
            short_long_df = short_long_df.reset_index(drop = True)
            
            for m in range(2*k):
                if short_long_df.iloc[m,0] == short_long_df.iloc[m,1]:
                    accurate_item += 1
        accuracy = accurate_item/(2*k*len(list(date_list)))
        print('accuracy for '+str(2*k)+' portfolio in Study Period '+str(i)+' is '+ str(accuracy))
    accu += accuracy
print(accu/13)    
"""
After we compared different K, we have found the maximum accuracy was achieved when K =10 for almost all study period.
Hence, we decided to go with K = 10 for our further portfolio construction.

"""
        
        
        