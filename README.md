# Stock Prediction using Neural Network
---
In this project, I want to implement LSTM to predict stock price on a minute-level frequency.
Basically, I want LSTM learn from a set window size of data (maybe stock return in the last ten minutes) so that it can be used to predict the next N-steps in the series.

This project is divided into five part:
1. data processing
2. feature selection
3. model building
4. model evaluation
5. backtesting

## Data Processing
### 1. Download minute-by-minute stock data from GTADB.

### 2. Select suitable stocks in time series(for version 2.0)
1. high realized volatility 
	- 5/15 minute frequency pct data. 
	- Implemented by `realized_volatility.py` and save to `F:/data/rv.h5`

2. high turnover rate 
	- `volume/market`

3. no large jump at opening
	- `open_price/last_day_close_price`
	- get rid of new stocks



## Feature selection
To test the effectiveness of LSTM on stock prediction, use some very simple feature to test the model first.

1. percent change
2. jump change between minutes
3. volume
4. high/low (amplitude)
5. pct with moving average
7. pct of the market and industry
8. pct of related stock
9. mean pct
10. realized volatility

### standardize the feature(core of this project)
why standardize?
- make training faster
- less likely stucking in local optima
- gradients less likely explode or becomes too small (more likely if features are big)

#### For changes:
Easy since there's limit (20), so use `Min-Max normalization` method

#### For volume:
1. Use z-score with mean and std from previous 5/10 day
2. Use quantile compared with history/recent_days



### Selection methods
- SelectKBest Algorithm, f_regression, F statistic

- Autocorrelation/cross-correlation

- sklearn

## Model building

1. target: next minute percent change (up,down or stay) from 9:31 to 11:30 and 1:01 to 14:45
2. input:  minute pct from 9:30 to 11:29 and 1:00 to 14:44




## My notes when building the project
different stock may perform differently, maybe more suitable to build different model for different category of stocks. (By doing unsupervised learning first)

