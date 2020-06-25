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


def download_Fundamentals(tickers_list, ClosePrices):
    # empty dictionary to append data
    val_stats_dict = {}
    fndmntl_dict = {}

    # loop each stocks
    for ticker in tqdm(tickers_list, desc="Parsing progress:", unit="tickers"):
        valuation_stats = si.get_stats(ticker)
        valuation_stats.columns = ["Valuation", "Current"]

        income_statement = si.get_income_statement(ticker).transpose()
        # income_statement.columns = income_statement.iloc[0]
        # income_statement = income_statement.drop(income_statement.index[0])

        balance_sheet = si.get_balance_sheet(ticker).transpose()
        # balance_sheet.columns = balance_sheet.iloc[0]
        # balance_sheet = balance_sheet.drop(balance_sheet.index[0])

        cash_flow_statement = si.get_cash_flow(ticker).transpose()
        # cash_flow_statement.columns = cash_flow_statement.iloc[0]
        # cash_flow_statement = cash_flow_statement.drop(cash_flow_statement.index[0])

        three_statements = income_statement.merge(balance_sheet, left_index=True, right_index=True, how='outer')
        three_statements = three_statements.merge(cash_flow_statement, left_index=True, right_index=True, how='outer')
        three_statements = three_statements.iloc[0]
        three_statements = three_statements.drop(three_statements.index[0])

        val_stats_dict[ticker] = valuation_stats
        fndmntl_dict[ticker] = three_statements

        # look up price for each date in three_statement dataframe. Calculate price change from 1 year ago from that date

    combined_fndmntl = pd.concat(fndmntl_dict, sort=False).reset_index(drop=False)
    combined_fndmntl.rename(columns={"level_0": "Ticker", "level_1": "Date"}, inplace=True)
    
    combined_valuation_stats = pd.concat(val_stats_dict, sort=False).reset_index(drop=False)
    del combined_valuation_stats['level_1']
    combined_valuation_stats.rename(columns={"level_0": "Ticker"}, inplace=True)


    return combined_fndmntl, combined_valuation_stats

if __name__ == "__main__":

    ALL_NZX = [
        'ABA', 'AFC', 'AFT', 'AIA', 'AIR', 'ALF', 'AOR', 'APL', 'ARB', 'ARG', 'ARV', 'ATM', 'AUG', 'AWF', 'BFG', 'BGI', 'BGP', 'BLT', 'CAV','CBD', 'CDI', 'CEN', 'CGF', 'CMO', 'CNU', 'CVT', 'DGL', 'EBO', 'ENS', 'ERD', 'EVO', 'FBU', 'FPH', 'FRE', 'FSF', 'FWL', 'GEN', 'GEO', 'GFL', 'GMT', 'GNE', 'GSH', 'GTK', 'GXH', 'HGH', 'HLG', 'IFT', 'IKE', 'IPL', 'JLG', 'KMD', 'KPG', 'MCK', 'MCY', 'MEE', 'MEL', 'MET', 'MFT', 'MGL', 'MMH', 'MOA', 'MPG', 'MWE', 'NPH', 'NTL', 'NWF', 'NZK', 'NZM', 'NZO', 'NZR', 'NZX', 'OCA', 'PCT', 'PEB', 'PFI', 'PGW', 'PIL', 'PLX', 'POT', 'PPH', 'PYS', 'QEX', 'RAK', 'RBD', 'RYM', 'SAN', 'SCL', 'SCT', 'SCY', 'SDL', 'SEK', 'SKC', 'SKL', 'SKO', 'SKT', 'SML', 'SNC', 'SPG', 'SPK', 'SPN', 'SPY', 'STU', 'SUM', 'TGG', 'THL', 'TLL', 'TLT', 'TPW', 'TRA', 'TRS', 'TRU', 'TWR', 'VCT', 'VGL', 'VHP', 'VTL', 'WDT', 'WHS', 'ZEL'
        ]
    ALL_INDX = ['^NZ50','^GSPC','^DJI','^VIX']
    ClosePrices = download_Prices([x+'.NZ' for x in ALL_NZX[:2]] + ALL_INDX)
    download_Fundamentals([x+'.NZ' for x in ALL_NZX[:2]], ClosePrices)
    print(f'DONE !!!!!!!!!')