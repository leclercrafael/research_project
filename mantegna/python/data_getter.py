import pandas as pd
import numpy as np
import requests
from io import StringIO
import yfinance as yf
import os

class Data_Getter:

    def __init__(self):
        self.tickers = []
        self.data = None

    def get_tickers(self, get_adr : bool = True) -> list :
        self.url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"} 
        self.get_adr = get_adr
        try:

            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()  # Vérifie si la requête a réussi


            tables = pd.read_html(StringIO(response.text))
            df = tables[0]
            tickers = df['Symbol'].tolist()

            if get_adr:
                top_50_adrs = ['TSM', 'ASML', 'SAP', 'INFY', 'SONY', 'STM', 'UMC', 'BABA', 'PDD', 'JD', 'BIDU', 'SE', 'SHOP', 'MELI', 'NVO', 'AZN', 'NVS', 'SNY', 'GSK', 'TAK', 'BNTX', 'TM', 'HMC', 'STLA', 'RACE', 'LI', 'SHEL', 'TTE', 'BP', 'PBR', 'EQNR', 'BHP', 'RIO', 'VALE', 'HSBC', 'RY', 'TD', 'HDB', 'UBS', 'MUFG', 'IX', 'BCS', 'DB', 'ING', 'SAN', 'BBD', 'UL', 'DEO', 'BUD', 'BTI']

                tickers += top_50_adrs

            self.tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            print(f"Succès : {len(tickers)} tickers récupérés.")
            return tickers

        except Exception as e:
            print(f"Erreur lors de la récupération : {e}")
            return []
        
    def get_financials(self, tickers : list, start_date : str, end_date : str) -> pd.DataFrame :

        self.start_date = start_date
        self.end_data = end_date

        if self.get_adr :
            self.filename = f"sp500+50adr_{start_date}_{end_date}.csv"
        else : 
            self.filename = f"sp500_{start_date}_{end_date}.csv"

        if os.path.exists('./csv/'+ self.filename):
            self.data = pd.read_csv("./csv/"+ self.filename, index_col=0, parse_dates=True)
        
            if not self.data.empty:
                    return self.data
             
        self.raw_data = yf.download(tickers=tickers, start=start_date, end=end_date, auto_adjust=True, threads=True)

        if 'Close' in self.raw_data.columns:
             self.data = self.raw_data['Close']
        else:
             self.data = self.raw_data

        self.data = self.data.dropna(axis=1, how='all')

        print(f"Sauvegarde des données dans {self.filename}...")
        self.data.to_csv('./csv/'+self.filename)
        
        return self.data
    
    def get_log_returns(self) -> pd.DataFrame:
         
        if self.data is None:
            print("Erreur: Aucune donnée chargée. Lancez get_financials d'abord.")
            return pd.DataFrame()
        
        if os.path.exists('./csv/log_returns_'+ self.filename):
            log_returns = pd.read_csv("./csv/log_returns_"+ self.filename, index_col=0, parse_dates=True)
            return log_returns
            
        log_returns = np.log(self.data / self.data.shift(1))
        log_returns = log_returns.dropna(axis=0, how='all').dropna(axis=1, how='all')

        print(f"Sauvegarde des données dans log_returns_{self.filename}...")
        log_returns.to_csv('./csv/log_returns_'+ self.filename)

        return log_returns





        