import pandas as pd
import numpy as np
import time
import os
import requests
from data_keys import *
# import returns

data_dir = 'data/'


def _resample_daily(data):
    """
    Resample data using linear interpolation.

    :param data: Pandas DataFrame or Series.
    :return: Resampled daily data.
    """
    return data.resample('D').interpolate(method='linear')


def _load_data(path):
    """
    Load a CSV-file with tab-separation, date-index is in first column
    and uses the MM/DD/YYYY.

    This is a simple wrapper for Pandas.read_csv().

    :param path: Path for the data-file.
    :return: Pandas DataFrame.
    """
    data = pd.read_csv(path,
                       sep="\t",
                       index_col=0,
                       parse_dates=True,
                       dayfirst=False)

    return data


def load_price_yahoo(ticker):
    """
    Load share-price data from a Yahoo CSV-file.

    Only retrieve the 'Close' and 'Adj Close' prices
    which are interpolated to daily values.

    The 'Close' price-data is adjusted for stock-splits.

    The 'Adj Close' price-data is adjusted for both
    stock-splits and dividends, so it corresponds to
    the Total Return.

    https://help.yahoo.com/kb/SLN2311.html

    :param ticker: Ticker-name for the data to load.
    :return: Pandas DataFrame with SHARE_PRICE and TOTAL_RETURN
    """

    # Path for the data-file to load.
    path = os.path.join(data_dir, ticker + " Share-Price (Yahoo).csv")

    # Read share-prices from file.
    price_raw = pd.read_csv(path,
                            index_col=0,
                            header=0,
                            sep=',',
                            parse_dates=[0],
                            dayfirst=False)

    # Rename columns.
    columns = \
        {
            'Adj Close': TOTAL_RETURN,
            'Close': SHARE_PRICE
        }
    price = price_raw.rename(columns=columns)

    # Select the columns we need.
    price = price[[TOTAL_RETURN, SHARE_PRICE]]

    # Interpolate to get prices for all days.
    price_daily = _resample_daily(price)

    return price_daily


def load_earnings_per_share(ticker, df, profit_margin=True):
    """
    Load the Earnings Per Share from a data-file and add it to the DataFrame.
    Also calculate the P/E ratio and profit margin.

    :param ticker:
        Name of the stock used in the filenames e.g. "WMT"

    :param df:
        Pandas DataFrame with SHARE_PRICE.

    :param profit_margin:
        Boolean whether to add the profit margin to the DataFrame.
        Requires that df already contains SALES_PER_SHARE.

    :return:
        None. Data is added to the `df` DataFrame.
    """

    # Load data.
    path = os.path.join(data_dir, ticker + " Earnings Per Share.txt")
    earnings_per_share = _load_data(path=path)

    # Add to the DataFrame (interpolated daily).
    df[EARNINGS_PER_SHARE] = _resample_daily(earnings_per_share)

    # Add valuation ratio to the DataFrame (daily).
    df[PE] = df[SHARE_PRICE] / df[EARNINGS_PER_SHARE]

    # Add profit margin to the DataFrame (daily).
    if profit_margin:
        df[PROFIT_MARGIN] = df[EARNINGS_PER_SHARE] / df[SALES_PER_SHARE]


def load_sales_per_share(ticker, df):
    """
    Load the Sales Per Share from a data-file and add it to the DataFrame.
    Also calculate the P/Sales ratio and one-year growth in Sales Per Share.

    :param ticker:
        Name of the stock used in the filenames e.g. "WMT"

    :param df:
        Pandas DataFrame with SHARE_PRICE.

    :return:
        None. Data is added to the `df` DataFrame.
    """

    # Load data.
    path = os.path.join(data_dir, ticker + " Sales Per Share.txt")
    sales_per_share = _load_data(path=path)

    # Add to the DataFrame (interpolated daily).
    df[SALES_PER_SHARE] = _resample_daily(sales_per_share)

    # # Add valuation ratio to the DataFrame (daily).
    df[PSALES] = df[SHARE_PRICE] / df[SALES_PER_SHARE]

    # # Add growth to the DataFrame (daily).
    df[SALES_GROWTH] = df[SALES_PER_SHARE].pct_change(periods=365, fill_method=None)


def load_book_value_per_share(ticker, df):
    """
    Load the Book-Value Per Share from a data-file and add it to the DataFrame.
    Also calculate the P/Book ratio.

    :param ticker:
        Name of the stock used in the filenames e.g. "WMT"

    :param df:
        Pandas DataFrame with SHARE_PRICE.

    :return:
        None. Data is added to the `df` DataFrame.
    """

    # Load data.
    path = os.path.join(data_dir, ticker + " Book-Value Per Share.txt")
    book_value_per_share = _load_data(path=path)

    # Add to the DataFrame (interpolated daily).
    df[BOOK_VALUE_PER_SHARE] = _resample_daily(book_value_per_share)

    # Add valuation ratio to the DataFrame (daily).
    df[PBOOK] = df[SHARE_PRICE] / df[BOOK_VALUE_PER_SHARE]


def load_dividend_TTM(ticker, df):
    """
    Load the Dividend Per Share TTM (Trailing Twelve Months) from a data-file and
    add it to the DataFrame. Also calculate some related valuation ratios.

    :param ticker:
        Name of the stock-index used in the filenames e.g. "S&P 500"

    :param df:
        Pandas DataFrame with SHARE_PRICE.

    :return:
        None. Data is added to the `df` DataFrame.
    """

    # Load data.
    path = os.path.join(data_dir, ticker + " Dividend Per Share TTM.txt")
    dividend_per_share_TTM = _load_data(path=path)

    # Add to the DataFrame (interpolated daily).
    df[DIVIDEND_TTM] = _resample_daily(dividend_per_share_TTM)

    # Add valuation ratios to the DataFrame (daily).
    df[PDIVIDEND] = df[SHARE_PRICE] / df[DIVIDEND_TTM]
    df[DIVIDEND_YIELD] = df[DIVIDEND_TTM] / df[SHARE_PRICE]


def load_stock_data(ticker, earnings=True, sales=True, book_value=True,
                    dividend_TTM=False):
    """
    Load data for a single stock from several different files
    and combine them into a single Pandas DataFrame.

    - Price is loaded from a Yahoo-file.
    - Other data is loaded from separate files.

    The Total Return is taken directly from the Yahoo price-data.
    Valuation ratios such as P/E and P/Sales are calculated daily
    from interpolated data.

    :param ticker:
        Name of the stock used in the filenames e.g. "WMT"

    :param earnings:
        Boolean whether to load data-file for Earnings Per Share.

    :param sales:
        Boolean whether to load data-file for Sales Per Share.

    :param book_value:
        Boolean whether to load data-file for Book-Value Per Share.

    :param dividend_TTM:
        Boolean whether to load data-file for Dividend Per Share TTM.

    :return: Pandas DataFrame with the data.
    """
    # Load the data-files.
    price_daily = load_price_yahoo(ticker=ticker)

    # Use the DataFrame for the price and add more data-columns to it.
    df = price_daily

    # Only keep the rows where the share-price is defined.
    df.dropna(subset=[SHARE_PRICE], inplace=True)

    # Load Sales Per Share data.
    if sales:
        load_sales_per_share(ticker=ticker, df=df)

    # Load Earnings Per Share data.
    # This needs the Sales Per Share data to calculate the profit margin.
    if earnings:
        load_earnings_per_share(ticker=ticker, df=df)

    # Load Book-Value Per Share data.
    if book_value:
        load_book_value_per_share(ticker=ticker, df=df)

    # Load Dividend Per Share TTM data.
    if dividend_TTM:
        load_dividend_TTM(ticker=ticker, df=df)

    return df