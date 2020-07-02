""" 
    guide for yahoo finance data scrapping:
    yfinance: https://algotrading101.com/learn/yfinance/
    yahoo_fin: https://algotrading101.com/learn/yahoo-finance-api/ , http://theautomatic.net/2018/01/25/coding-yahoo_fin-package/ 
    freemium source:RapidAPI: https://rapidapi.com/apidojo/api/yahoo-finance1/endpoints
"""
"""
    - use return instead of price difference as dependent variable
    - compute graham number
    - add explanatory variables: OCR, business confidence, GDP, inflation, interest rate, 
    - use PE ratio, sharpe ratio in Timbre, if exists
"""
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin import stock_info as si
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from tqdm import tqdm


def download_Prices(tickers_list):
    # Download prices for the last 10 years
    data = yf.download(
        tickers=tickers_list,
        start=datetime(
            datetime.now().year - 10, datetime.now().month, datetime.now().day
        ).date(),
        interval="1d",  # 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        group_by="column",  #'ticker', 'column'
        auto_adjust=True,
        threads=True,
    )

    # Get close price. Replace nan value value by previous value
    ClosePrices = data["Close"].reset_index()
    ClosePrices.fillna(method="ffill", inplace=True)

    # pivot
    ClosePrices = ClosePrices.melt(id_vars=["Date"])
    ClosePrices.rename(columns={"variable": "Ticker", "value": "Price"}, inplace=True)

    # lag and lead price and return
    ClosePrices["Prev250day"] = ClosePrices.groupby(["Ticker"])["Date"].shift(250)
    ClosePrices["Prev250dayPrice"] = ClosePrices.groupby(["Ticker"])["Price"].shift(250)    
    ClosePrices["Prev250dayReturn"] = ClosePrices["Price"]/ClosePrices["Prev250dayPrice"]-1

    ClosePrices["Next250day"] = ClosePrices.groupby(["Ticker"])["Date"].shift(-250)
    ClosePrices["Next250dayPrice"] = ClosePrices.groupby(["Ticker"])["Price"].shift(-250)
    ClosePrices["Next250dayReturn"] = ClosePrices["Next250dayPrice"]/ClosePrices["Price"]-1

    # to csv
    ClosePrices.to_csv("_stock_prices.csv", index=False, date_format='%d/%m/%Y')

    return ClosePrices


def Read_download_Prices():
    # dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
    df = pd.read_csv("_stock_prices.csv", parse_dates=['Date','Prev250day','Next250day'])#, date_parser=dateparse)

    return df


def download_Fundamentals(tickers_list, ClosePrices):
    # empty dictionary to append data
    val_stats_dict = {}
    fndmntl_dict = {}

    # loop each stocks
    for ticker in tqdm(tickers_list, desc="Parsing progress:", unit="tickers"):
        valuation_stats = si.get_stats(ticker)
        valuation_stats.columns = ["Valuation", "Current"]

        income_statement = (
            si.get_income_statement(ticker).set_index("Breakdown").transpose()
        )

        balance_sheet = si.get_balance_sheet(ticker).set_index("Breakdown").transpose()

        cash_flow_statement = (
            si.get_cash_flow(ticker).set_index("Breakdown").transpose()
        )

        three_statements = income_statement.merge(
            balance_sheet, left_index=True, right_index=True, how="outer"
        )
        three_statements = three_statements.merge(
            cash_flow_statement, left_index=True, right_index=True, how="outer"
        )

        val_stats_dict[ticker] = valuation_stats
        fndmntl_dict[ticker] = three_statements

    # combine three statements
    combined_three_statements = pd.concat(fndmntl_dict, sort=False).reset_index(
        drop=False
    )
    combined_three_statements.rename(
        columns={"level_0": "Ticker", "level_1": "Date"}, inplace=True
    )
    combined_three_statements = combined_three_statements[combined_three_statements['Date'] != 'ttm'].reset_index(drop=True)
    combined_three_statements["Date"] = pd.to_datetime(
        combined_three_statements["Date"], format="%m/%d/%Y"
    )

    # combine statistics valuation
    combined_valuation_stats = pd.concat(val_stats_dict, sort=False).reset_index(
        drop=False
    )
    del combined_valuation_stats["level_1"]
    combined_valuation_stats.rename(columns={"level_0": "Ticker"}, inplace=True)

    return combined_three_statements, combined_valuation_stats


# lookup future share price from each statement date. 
# TODO would like to do the same for each valuation stats as well, but NZ stocks only have most recent data. Unlike US stocks, have historical data
def Join_Fndmntl_w_Price(combined_three_statements, ClosePrices, ALL_INDX):
    # extract unique date in combine three statements & close price >>> sort date columns both dataframe >>> duplicate date column in close price df, because merge_asof will remove date column in close price >>> merge_asof and rename duplicate date column to trade date
    dt_combine_three_statements = combined_three_statements['Date'].drop_duplicates().sort_values().reset_index(drop=True)
    ClosePrices['Duplicate_Date'] = ClosePrices['Date']
    dt_ClosePrices = ClosePrices[['Date','Duplicate_Date']].drop_duplicates().sort_values(['Date']).reset_index(drop=True)    
    df_lookup_tradedate = pd.merge_asof(
        dt_combine_three_statements,
        dt_ClosePrices,
        on=["Date"],
        direction="backward",
        allow_exact_matches=True
    )

    # look up price for each date in combined_three_statements dataframe
    combined_three_statements = combined_three_statements.merge(df_lookup_tradedate, on='Date', how='left')
    combined_three_statements.rename(columns={"Duplicate_Date": "Trade Date"}, inplace=True)
    combined_three_statements = combined_three_statements.merge(
        ClosePrices[['Date', 'Ticker', 'Next250dayReturn']],
        left_on=['Ticker','Trade Date'],
        right_on=['Ticker','Date'],
        how='inner'
        ).drop('Date_y', axis=1)
    combined_three_statements.rename(columns={"Date_x": "Date"}, inplace=True)

    # look up past index return for each date.
    for idx in ALL_INDX:
        Idx_ClosePrices = ClosePrices[ClosePrices['Ticker']==idx].copy()
        # if nzx 50 index then use future return
        if idx == '^NZ50':
            Idx_ClosePrices.rename(columns={'Next250dayReturn':'Next250dayReturn'+idx}, inplace=True)

            combined_three_statements = combined_three_statements.merge(
                Idx_ClosePrices[['Date', 'Ticker', 'Next250dayReturn'+idx]],
                left_on=['Trade Date'],
                right_on=['Date'],
                how='inner'
                ).drop(['Date_y','Ticker_y'], axis=1)
        # all other index, use past return
        else:            
            Idx_ClosePrices.rename(columns={'Prev250dayReturn':'Prev250dayReturn'+idx}, inplace=True)

            combined_three_statements = combined_three_statements.merge(
                Idx_ClosePrices[['Date', 'Ticker', 'Prev250dayReturn'+idx]],
                left_on=['Trade Date'],
                right_on=['Date'],
                how='inner'
                ).drop(['Date_y','Ticker_y'], axis=1)
        
        combined_three_statements.rename(columns={"Ticker_x":'Ticker',"Date_x": "Date"}, inplace=True)

    combined_three_statements['BeatIndex'] = combined_three_statements['Next250dayReturn'] > combined_three_statements['Next250dayReturn^NZ50']

    # clean - character; Nan
    combined_three_statements.replace('-', 0, inplace=True)
    combined_three_statements.fillna(0, inplace=True)

    # reorder columns so that explanatory variable are on the right hand side of dataframe
    non_train_col = ['Ticker','Date','Trade Date','Next250dayReturn','Next250dayReturn^NZ50','BeatIndex']
    list_trainer_col = [x for x in combined_three_statements.columns.tolist() if x not in non_train_col]
    combined_three_statements = combined_three_statements[non_train_col + list_trainer_col]
    
    # to csv
    combined_three_statements.to_csv("_three_statements.csv", index=False, date_format='%d/%m/%Y')
    
    return combined_three_statements


def Read_combined_Fundamental():
    # dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
    df = pd.read_csv("_three_statements.csv", parse_dates=['Date','Trade Date'])    #, date_parser=dateparse)

    return df


def Learn(combined_three_statements, ALL_INDX):
    non_train_col = ['Ticker','Date','Trade Date','Next250dayReturn','Next250dayReturn^NZ50','BeatIndex']
    list_trainer_col = [x for x in combined_three_statements.columns.tolist() if x not in non_train_col]

    # Generate the train set and test set by randomly splitting the dataset
    X = combined_three_statements.iloc[:,-len(list_trainer_col):].values
    Y = combined_three_statements['BeatIndex'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Instantiate a RandomForestClassifier with 100 trees, then fit it to the training data
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, Y_train)

    # Generate the predictions, then print test set accuracy and precision
    Y_predict = clf.predict(X_test)
    print("Classifier performance\n", "=" * 20)
    print(f"Accuracy score: {clf.score(X_test, Y_test): .2f}")
    print(f"Precision score: {precision_score(Y_test, Y_predict): .2f}")

    print("a")


if __name__ == "__main__":

    ALL_NZX = [
        "ABA",
        "AFC",
        "AFT",
        "AIA",
        "AIR",
        "ALF",
        "AOR",
        "APL",
        "ARB",
        "ARG",
        "ARV",
        "ATM",
        "AUG",
        "AWF",
        "BFG",
        "BGI",
        "BGP",
        "BLT",
        "CAV",
        "CBD",
        "CDI",
        "CEN",
        "CGF",
        "CMO",
        "CNU",
        "CVT",
        "DGL",
        "EBO",
        "ENS",
        "ERD",
        "EVO",
        "FBU",
        "FPH",
        "FRE",
        "FSF",
        "FWL",
        "GEN",
        "GEO",
        "GFL",
        "GMT",
        "GNE",
        "GSH",
        "GTK",
        "GXH",
        "HGH",
        "HLG",
        "IFT",
        "IKE",
        "IPL",
        "JLG",
        "KMD",
        "KPG",
        "MCK",
        "MCY",
        "MEE",
        "MEL",
        "MET",
        "MFT",
        "MGL",
        "MMH",
        "MOA",
        "MPG",
        "MWE",
        "NPH",
        "NTL",
        "NWF",
        "NZK",
        "NZM",
        "NZO",
        "NZR",
        "NZX",
        "OCA",
        "PCT",
        "PEB",
        "PFI",
        "PGW",
        "PIL",
        "PLX",
        "POT",
        "PPH",
        "PYS",
        "QEX",
        "RAK",
        "RBD",
        "RYM",
        "SAN",
        "SCL",
        "SCT",
        "SCY",
        "SDL",
        "SEK",
        "SKC",
        "SKL",
        "SKO",
        "SKT",
        "SML",
        "SNC",
        "SPG",
        "SPK",
        "SPN",
        "SPY",
        "STU",
        "SUM",
        "TGG",
        "THL",
        "TLL",
        "TLT",
        "TPW",
        "TRA",
        "TRS",
        "TRU",
        "TWR",
        "VCT",
        "VGL",
        "VHP",
        "VTL",
        "WDT",
        "WHS",
        "ZEL",
    ]
    ALL_INDX = ["^NZ50", "^GSPC", "^DJI", "^VIX"]

    #. 
    # ClosePrices = download_Prices([x+'.NZ' for x in ALL_NZX[:2]] + ALL_INDX)
    ClosePrices = Read_download_Prices()

    #.
    # combined_three_statements, combined_valuation_stats = download_Fundamentals([x+'.NZ' for x in ALL_NZX], ClosePrices)    

    #.
    # combined_three_statements = Join_Fndmntl_w_Price(combined_three_statements, ClosePrices, ALL_INDX)
    combined_three_statements = Read_combined_Fundamental()

    #.
    Learn(combined_three_statements, ALL_INDX)


    print(f"DONE !!!!!!!!!")

