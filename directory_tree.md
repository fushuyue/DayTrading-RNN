## Everyday procedural
### Pretrade data
#### Spread `pretrade_spread.py`

#### Realized volatility & range `pretrade_vol.py`

#### Select suitable stocks `choose_stock_tick_v2.py`

#### Normalization data
1. last close price
2. 30 days average trading volume 
3. 30 days average best bid ask size

### For backtesting
#### 1. extract tick data: `extract_tick.py`
file saved to : `filename = 'g:/tick_data/tickdata_'+ d + '.h5'`

#### 2. prepare features: `prepare_tick_2015.py`
file needed:
- original data: `filename = 'g:/tick_data/tickdata_'+ d + '.h5'`
- realized volatility: `g:/pretrade/rv_pivot_+ d + .h5`
- realized range: `g:/pretrade/range_pivot_+ d + .h5`
- data for normalization: `g:/pretrade/pre_trade_bd.h5` & `g:/pretrade/pre_trade_vol.h5`
- selected stocks: `g:/pretrade/stock_list_ + d + .csv`

file saved to: `'g:/lstm/lstm_tick_' + d + '.h5'` with four earray



### Daily Updates 
1. tick data: update_tick.py
2. calculate pretrade data: update_tick_pretrade.py
3. upload pretrade data into database: prepare_twap_v2.py
4. calculate and update features: update_feature.py

### Version control
1. prepare_feature_v5.py: last version currently. Fix bugs on trading volume normalization
2. prepare_feature_v5_v2.py: add first 5 minutes into training set.
