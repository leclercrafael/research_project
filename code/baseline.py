import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf

from data_getter import Data_getter

start_date = '2004-01-01'
end_date = '2024-01-01'

#Import data 
data_getter = Data_getter()

#Import tickers
sp500_tickers = data_getter.get_tickers()

#Import data (Adj Close)
data = data_getter.get_financials(tickers=sp500_tickers, start_date=start_date, end_date=end_date)

#Import log returns
log_returns = data_getter.get_log_returns()

tickers_sample = sp500_tickers[:50] 
print(tickers_sample)



# print("Téléchargement des données...")
# data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

