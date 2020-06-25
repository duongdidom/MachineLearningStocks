""" 
    guide for yahoo finance data scrapping:
    yfinance: https://algotrading101.com/learn/yfinance/
    yahoo_fin: https://algotrading101.com/learn/yahoo-finance-api/ , http://theautomatic.net/2018/01/25/coding-yahoo_fin-package/ 
    freemium source:RapidAPI: https://rapidapi.com/apidojo/api/yahoo-finance1/endpoints
"""
import pandas as pd
import yfinance as yf
from yahoo_fin import stock_info as si

from tqdm import tqdm

def download_Prices(tickers_list):
    # Download
    data = yf.download(
        tickers=tickers_list,
        period='max', 
        interval='1d', #1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        group_by = 'column',    #'ticker' or 'column'
        auto_adjust=True,
        threads=True
        )

    # Get close price. Replace nan value value by previous value
    ClosePrices = data['Close'].reset_index()
    ClosePrices.fillna(method='ffill', inplace=True)

    # to csv
    ClosePrices.to_csv('_stock_prices.csv', index=False)
    
    return ClosePrices


def download_Fundamentals(tickers_list):
    # empty dictionary to append data
    fndmntl_dict = {}

    for ticker in tqdm(tickers_list, desc="Parsing progress:", unit="tickers"):
        valuation_stats = si.get_stats_valuation(ticker)
        valuation_stats.columns = ["Valuation", "Current"]


        ticker_object = yf.Ticker(ticker)

        #convert info() output from dictionary to dataframe
        temp_df = pd.DataFrame.from_dict(ticker_object.info, orient="index")
        temp_df.reset_index(inplace=True)
        temp_df.columns = ["Attribute", "Recent"]
        
        # add (ticker, dataframe) to main dictionary
        fndmntl_dict[ticker] = temp_df

    combined_data = pd.concat(fndmntl_dict)
    combined_data.reset_index(drop=True)
    combined_data.columns = ["Ticker", "Attribute", "Recent"]

if __name__ == "__main__":

    ALL_NZX = [
        'ABA', 'AFC', 'AFT', 'AIA', 'AIR', 'ALF', 'AOR', 'APL', 'ARB', 'ARG', 'ARV', 'ATM', 'AUG', 'AWF', 'BFG', 'BGI', 'BGP', 'BLT', 'CAV','CBD', 'CDI', 'CEN', 'CGF', 'CMO', 'CNU', 'CVT', 'DGL', 'EBO', 'ENS', 'ERD', 'EVO', 'FBU', 'FPH', 'FRE', 'FSF', 'FWL', 'GEN', 'GEO', 'GFL', 'GMT', 'GNE', 'GSH', 'GTK', 'GXH', 'HGH', 'HLG', 'IFT', 'IKE', 'IPL', 'JLG', 'KMD', 'KPG', 'MCK', 'MCY', 'MEE', 'MEL', 'MET', 'MFT', 'MGL', 'MMH', 'MOA', 'MPG', 'MWE', 'NPH', 'NTL', 'NWF', 'NZK', 'NZM', 'NZO', 'NZR', 'NZX', 'OCA', 'PCT', 'PEB', 'PFI', 'PGW', 'PIL', 'PLX', 'POT', 'PPH', 'PYS', 'QEX', 'RAK', 'RBD', 'RYM', 'SAN', 'SCL', 'SCT', 'SCY', 'SDL', 'SEK', 'SKC', 'SKL', 'SKO', 'SKT', 'SML', 'SNC', 'SPG', 'SPK', 'SPN', 'SPY', 'STU', 'SUM', 'TGG', 'THL', 'TLL', 'TLT', 'TPW', 'TRA', 'TRS', 'TRU', 'TWR', 'VCT', 'VGL', 'VHP', 'VTL', 'WDT', 'WHS', 'ZEL'
        ]
    ALL_INDX = ['^NZ50','^GSPC','^DJI','^VIX']
    # download_Prices([x+'.NZ' for x in ALL_NZX[:2]] + ALL_INDX)
    download_Fundamentals([x+'.NZ' for x in ALL_NZX[:2]])
    print(f'DONE !!!!!!!!!')