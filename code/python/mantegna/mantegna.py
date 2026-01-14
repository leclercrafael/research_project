import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

import yfinance as yf

from data_getter import Data_Getter



class Mantegna():

    def __init__(self, start_date : str = '2004-01-01', end_date : str = '2024-01-01'):
        self.start_date = start_date
        self.end_date = end_date

    def plot_static_mantegna(self) -> None : 
        '''
        Start/End : intervall of time for financials data gathering
        '''
        




