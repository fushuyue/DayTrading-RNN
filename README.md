# Stock Prediction using Neural Network
---
In this project, I want to implement LSTM to predict stock price on a minute-level frequency.
Basically, I want LSTM model to learn from a set window size of data (maybe stock return in the last ten minutes) so that it can be used to predict the next N-steps in the series.

This project is divided into five part:
1. data processing
2. feature selection
3. model building
4. model evaluation
5. backtesting

## Data Processing
Download minute-by-minute stock data from GTADB.

Choose the suitable stock data 
- average daily turnover rate for last week
- average daily volatility for last week
- exclude stocks that opening price reach daily limit


## Feature selection
To test the effectiveness of LSTM on stock prediction, use some very simple feature to test the model first.



### test methods
greedy forward selection using cross validation

SelectKBest Algorithm, f_regression, F statistic

Autocorrelation/cross-correlation

compare the prediction accuracies of SVM algorithm and MART (a decision tree based boosting algorithm



## some thoughts 
what should be the target?
next one/five/fifteen minute's percent change?
- use one minute level feature to predict next 15 mintues pct

how to deal with opening information?
opening jump
- simple answer: do not choose them

how to deal with `time` feature
opening and closing has more information than other trading times

different stock may perform differently, maybe more suitable to build different model for different category of stocks. (By doing unsupervised learning first)
