from itertools import chain, combinations
import simfin as sf
from simfin.names import (TOTAL_RETURN, CLOSE, VOLUME, TICKER,
                          PSALES, DATE)
from simfin.utils import BDAYS_PER_YEAR
import numpy as np


def one_period_returns(prices, future):
    """

    Этот метод считает доходность по следующей формуле:

    R_t = (X_t - X_{t-1}) / X_{t-1}

    Calculate the one-period return for the given share-prices.
    
    Note that these have 1.0 added so e.g. 1.05 means a one-period
    gain of 5% and 0.98 means a -2% loss.

    :param prices:
        Pandas DataFrame with e.g. daily stock-prices.
        
    :param future:
        Boolean whether to calculate the future returns (True)
        or the past returns (False).
        
    :return:
        Pandas DataFrame with one-period returns.
    """
    # One-period returns plus 1.
    # percentage change
    rets = prices.pct_change(periods=1) + 1.0
    
    # Shift 1 time-step if we want future instead of past returns.
    if future:
        rets = rets.shift(-1)

    return rets

def get_bad_tickers(daily_returns_all, all_tickers, df_daily_prices):


    # Find tickers whose median daily trading market-cap < 1e6
    daily_trade_mcap = df_daily_prices[CLOSE] * df_daily_prices[VOLUME]
    mask = (daily_trade_mcap.groupby(level=0).median() < 1e7)
    bad_tickers1 = mask[mask].reset_index()[TICKER].unique()

    # Find tickers whose max daily returns > 100%
    mask2 = (daily_returns_all > 2.0)
    mask2 = (np.sum(mask2) >= 1) # если есть хотя бы один такой день, то убираем
    bad_tickers2 = mask2[mask2].index.to_list()

    # Find tickers which have too little data, so that more than 20%
    # of the rows are NaN (Not-a-Number).
    mask3 = (daily_returns_all.isna().sum(axis=0) > 0.2 * len(daily_returns_all))
    bad_tickers3 = mask3[mask3].index.to_list()

    # Find tickers that end with '_old'.
    # These stocks have been delisted for some reason.
    bad_tickers4 = [ticker for ticker in all_tickers[1:]
                    if ticker.endswith('_old')]

    # Tickers that we know have problematic / erroneous data.
    bad_tickers5= ['FCAUS']


    # Concatenate the different bad tickers we have found.
    bad_tickers = np.unique(np.concatenate([bad_tickers1, bad_tickers2,
                                            bad_tickers3, bad_tickers4,
                                            bad_tickers5]))
    
    return bad_tickers

def prepare_data(df_daily_prices):
    # Use the daily "Total Return" series which is the stock-price
    # adjusted for stock-splits and reinvestment of dividends.
    # This is a Pandas DataFrame in matrix-form where the rows are
    # time-steps and the columns are for the individual stock-tickers.
    daily_prices = df_daily_prices[TOTAL_RETURN].unstack().T

    # Remove rows that have very little data. Sometimes this dataset
    # has "phantom" data-points for a few stocks e.g. on weekends.
    num_stocks = len(daily_prices.columns)
    daily_prices = daily_prices.dropna(thresh=int(0.1 * num_stocks))

    # Remove the first row because sometimes it is incomplete.
    daily_prices = daily_prices.iloc[:, 1:]

    # Daily stock-returns calculated from the "Total Return".
    # We could have used SimFin's function hub.returns() but
    # this code makes it easier for you to use another data-source.
    # This is a Pandas DataFrame in matrix-form where the rows are
    # time-steps and the columns are for the individual tickers.
    daily_returns_all = one_period_returns(prices=daily_prices, future=True)

    # Remove empty rows (this should only be the first row).
    daily_returns_all = daily_returns_all.dropna(how='all')

    # All available stock-tickers.
    all_tickers = daily_prices.columns.to_list()

    bad_tickers = get_bad_tickers(daily_returns_all, all_tickers, df_daily_prices)

    # These are the valid stock-tickers we will be using.
    valid_tickers = [x for x in all_tickers if x not in bad_tickers] 

    # IMPORTANT!
    # This forward- and backward-fills the daily share-prices
    # so they are available for all the same dates. Otherwise
    # when we create a portfolio of many random stocks, we would
    # have to limit the period to the range of dates that all stocks
    # have in common. For large portfolios of e.g. 300 stocks they
    # would often only have common data-periods for a few years.
    # When the missing share-prices are filled like this, it
    # corresponds to investing that part of the portfolio in cash.
    # This does not matter for the Buy&Hold, Threshold, Adaptive
    # and Adaptive+ portfolios, but it may affect the Rebalanced
    # portfolio. You can easily disable this filling and only use
    # the original share-price data in all the experiments below,
    # simply by changing to `if False` instead.
    if True:
        # Forward- and backward-fill the missing share-prices.
        daily_prices = daily_prices.ffill().bfill()
        
        # Re-calculate the daily returns with the filled share-prices.
        # This creates 0% returns for the filled data, which basically
        # just corresponds to a cash-position.
        daily_returns_all = \
            one_period_returns(prices=daily_prices, future=True)
        
    # Only use data for the valid stock-tickers.
    daily_returns = daily_returns_all[valid_tickers]

    # убираем последню строчку, так как там только наны
    daily_returns = daily_returns.iloc[:-1]    

    return daily_returns
