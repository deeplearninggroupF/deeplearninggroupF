# deeplearninggroupF
LSTM For Stock Market Prediction

Reference paper: 
Deep learning with long short-term memory networks for financial market predictions by Thomas Fischer, Christopher Krauss

1_Data_Collection.ipynb

We collected our data using Bloomberg terminal, including the daily price and daily volume of all S&P 500 constitutents from 11/2002-10/2018.
For sector performance, we use 11 sector index funds'price (S5FINL, S5INFT, S5RLST, S5UTIL, S5ENRS, S5MATR, S5HLTH, S5COND, S5CONS, S5INDU, S5TELS)
The code shows how we merge, organize and clean different Excel files together to create one single CSV file which contains cross-sectional time series information of all S&P 500 constituents and sectors.

2_Data_Preparation.ipynb

The code shows how to calculate the daily return of stock price, stock volume and sector price.
Also, the code shows how to generate 1/0 target based on cross-sectional median of SP 500 constituents' daily return
Furthermore, the code shows how to seperate our dataset into 13 study period and normalize all three features using the mean and standard deviation of training set.
Finally, the code shows how to split our dataset into training set and testing set within each study period and prepare to feed our code into our LSTM model.

3_LSTM_Model.ipynb

The code shows how we use Tensorflow to implement LSTM model to predict stock market return.
We update the weights of our LSTM model from last study period for next study period and we output our prediciton results in all study period as a single CSV file.

4_Accuracy_Backtesting.ipynb

The code shows how we analyze the overall accuracy and accuracy for each sector of our LSTM model.
Also, the code shows how we construct and backtest three strategies (130-30, market neutral, long only) prior to transaction cost.
Finally, the code takes consideration of transaction cost and backtest three strategies again. 
