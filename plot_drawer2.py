import pandas as pd
import numpy as np
import time
import os
import requests
from data_keys import *
import matplotlib.pyplot as plt
from forecast_model import ForecastModel
from matplotlib.ticker import FuncFormatter


def pct_formatter(num_decimals=0):
    """Percentage-formatter used in plotting."""
    return FuncFormatter(lambda x, _: '{:.{}%}'.format(x, num_decimals))

def annualized_returns(series, years):
    """
    Calculate the annualized returns for all possible
    periods of the given number of years.

    For example, given the Total Return of a stock we want
    to know the annualized returns of all holding-periods
    of 10 years.

    :param series:
        Pandas series e.g. with the Total Return of a stock.
        Assumed to be daily data.
    :param years:
        Number of years in each period.
    :return:
        Pandas series of same length as the input series. Each
        day has the annualized return of the period starting
        that day and for the given number of years. The end of
        the series has NA for the given number of years.
    """

    # Number of days to shift data. All years have 365 days
    # except leap-years which have 366 and occur every 4th year.
    # So on average a year has 365.25 days.
    days = int(years * 365.25)

    # Calculate annualized returns for all periods of this length.
    # Note: It is important we have daily (interpolated) data,
    # otherwise the series.shift(365) would shift much more than
    # a year, if the data only contains e.g. 250 days per year.
    ann_return = (series.shift(-days) / series) ** (1 / years) - 1.0

    return ann_return


def prepare_ann_returns(df, years, key=PSALES, subtract=None):
    """
    Prepare annualized returns e.g. for making a scatter-plot.
    The x-axis is given by the key (e.g. PSALES) and the y-axis
    would be the annualized returns.

    :param df:
        Pandas DataFrame with columns named key and TOTAL_RETURN.
    :param years:
        Number of years for annualized returns.
    :param key:
        Name of the data-column for x-axis e.g. PSALES or PBOOK.
    :param subtract:
        Pandas Series to be subtracted from ann-returns
        to adjust for e.g. growth in sales-per-share.
    :return:
        (x, y) Pandas Series with key and adjusted ANN_RETURN.
    """

    # Create a new data-frame so we don't modify the original.
    # We basically just use this to sync the data we are
    # interested in for the common dates and avoid NA-data.
    df2 = pd.DataFrame()

    # Copy the key-data e.g. PSALES.
    df2[key] = df[key]

    # Calculate all annualized returns for all periods of
    # the given number of years using the Total Return.
    ann_return = annualized_returns(series=df[TOTAL_RETURN], years=years)

    if subtract is None:
        # Add the ann-returns to the new data-frame.
        df2[ANN_RETURN] = ann_return
    else:
        # Calculate all annualized returns for the series
        # that must be subtracted e.g. sales-per-share.
        ann_return_subtract = annualized_returns(series=subtract, years=years)

        # Subtract the ann. returns for the total return
        # and the adjustment (e.g. sales-per-share).
        # Then add the result to the new data-frame.
        df2[ANN_RETURN] = ann_return - ann_return_subtract

    # Drop all rows with NA.
    df2.dropna(axis=0, how='any', inplace=True)

    # Retrieve the relevant data.
    x = df2[key]
    y = df2[ANN_RETURN]

    return x, y


def plot_ann_returns(ticker, df, years, years_range=0,
                     dividend_yield=None, sales_growth=None,
                     psales=None, psales_min=None, psales_max=None,
                     ax=None, print_stats=False):
    """
    Create a plot with the actual historical returns showing
    the P/Sales ratios vs. future Annualized Returns.
    Overlay this plot with the estimated mean and std.dev.
    from the mathematical forecasting model.
    
    The optional params are taken from the DataFrame `df`
    if not supplied. This allows you to override some or
    all of the data used in the forecasting model e.g.
    to change assumptions about future sales-growth.
    
    :param ticker: String with ticker for the stock or index.
    :param df: Pandas DataFrame.
    :param years: Number of investment years.
    :param years_range:
        If > 0 then plot the mean ann. returns between
        years - years_range and years + years_range.
    :param dividend_yield: (Optional) Array with dividend yields.
    :param sales_growth: (Optional) Array with one-year sales growth.
    :param psales: (Optional) Array with P/Sales ratios.
    :param psales_min: (Optional) Min P/Sales for plotting curves.
    :param psales_max: (Optional) Max P/Sales for plotting curves.
    :param ax: (Optional) Matplotlib Axis object for the plot.
    :return: None
    """
    
    # Select only the data we need.
    df2 = df[[TOTAL_RETURN, DIVIDEND_YIELD, SALES_GROWTH, PSALES]]

    # Remove rows for which there is missing data.
    df2 = df2.dropna()
    
    # Create a new plot if no plotting-axis is supplied.
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)

    # Part of the title for the data's date-range.
    start_date, end_date = df2.index[[0, -1]]
    title_dates = "({0}-{1})".format(start_date.year, end_date.year)

    # # Get the actual ann. returns from the historic data.
    # if years_range > 0:
    #     # Use the mean. ann. returns between [min_years, max_years].
    #     min_years = years - years_range
    #     if min_years < 1:
    #         min_years = 1
    #     max_years = years + years_range

    #     # Get the mean ann.returns from the historic data.
    #     x, y = prepare_mean_ann_returns(df=df2,
    #                                     min_years=min_years,
    #                                     max_years=max_years,
    #                                     key=PSALES)

    #     # First part of the plot-title.
    #     title1 = "[{0}] {1}-{2} Year Mean Ann. Return {3}".format(ticker, min_years, max_years, title_dates)
        
    #     # Label for the scatter-plot of actual returns.
    #     label_returns = "Actual Returns (Mean)"
    # else:

    # Get the ann.returns from the historic data.
    x, y = prepare_ann_returns(df=df2, years=years, key=PSALES)

    # First part of the plot-title.
    title1 = "[{0}] {1}-Year Ann. Return {2}".format(ticker, years, title_dates)
    
    # Label for the scatter-plot of actual returns.
    label_returns = "Actual Returns"

    # Get Dividend Yield if none provided.
    if dividend_yield is None:
        dividend_yield = df2[DIVIDEND_YIELD]

    # Get Sales Growth if none provided.
    if sales_growth is None:
        sales_growth = df2[SALES_GROWTH]

    # Get P/Sales if none provided.
    if psales is None:
        psales = df2[PSALES]

    # Get min P/Sales for plotting if none provided.
    if psales_min is None:
        psales_min = np.min(psales)
    
    # Get max P/Sales for plotting if none provided.
    if psales_max is None:
        psales_max = np.max(psales)
        
    # Create the forecasting model and fit it to the data.
    model = ForecastModel(dividend_yield=dividend_yield,
                          sales_growth=sales_growth,
                          psales=psales, years=years)

    # Evenly spaced P/Sales ratios between historic min and max.
    psales_t = np.linspace(start=psales_min, stop=psales_max, num=100)

    # Use the model to forecast the mean and std ann.returns.
    mean, std = model.forecast(psales_t=psales_t)

    # Plot the mean ann.return with the R^2 for how well
    # it fits the actual ann.return.
    R_squared = model.R_squared(psales_t=x, ann_rets=y)
    label = "Forecast Mean (R^2 = {0:.2f})".format(R_squared)
    ax.plot(psales_t, mean, color="black", label=label)
    
    # Plot one standard deviation.
    color = "green"
    alpha = 0.3
    # Plot lines below and above mean.
    ax.plot(psales_t, mean-std, color=color,
            label="Forecast Mean $\pm$ 1 Std.Dev.")
    ax.plot(psales_t, mean+std, color=color)
    # Fill the areas.
    ax.fill_between(psales_t, mean+std, mean-std,
                    color=color, edgecolor=color, alpha=alpha)
    
    # Plot two standard deviations.
    color = "red"
    alpha = 0.1
    # Plot lines below and above mean.
    ax.plot(psales_t, mean-2*std, color=color,
            label="Forecast Mean $\pm$ 2 Std.Dev.")
    ax.plot(psales_t, mean+2*std, color=color) 
    # Fill the areas.
    ax.fill_between(psales_t, mean-std, mean-2*std,
                    color=color, edgecolor=color, alpha=alpha)
    ax.fill_between(psales_t, mean+std, mean+2*std,
                    color=color, edgecolor=color, alpha=alpha)

    # Scatter-plot with the actual P/Sales vs. Ann.Returns.
    # Each dot is colored according to its date (array-position).
    # The dots are rasterized (turned into pixels) to save space
    # when saving to vectorized graphics-file.
    n = len(x)
    c = np.arange(n) / n
    ax.scatter(x, y, marker='o', c=c, cmap='plasma',
               label=label_returns, rasterized=True)

    # Plot mean of Ann.Returns. as horizontal dashed line.
    y_mean = np.mean(y)
    label = 'Actual Mean = {0:.1%}'.format(y_mean)
    ax.axhline(y=y_mean, color="black", linestyle=":", label=label)

    # Show the labels for what we have just plotted.
    ax.legend()

    # Create plot-title.
    # Second part of the title. Formula for mean ann. return.
    msg = "E[Ann Return] = {0:.2f} / (P/Sales ^ (1/{1})) - 1"
    title2 = msg.format(model.a, years)
    # Third part of the title. Formula for std.dev. ann. return.
    msg = "Std[Ann Return] = {0:.3f} / (P/Sales ^ (1/{1}))"
    title3 = msg.format(model.b, years)
    # Combine and set the plot-title.
    title = "\n".join([title1, title2, title3])
    ax.set_title(title)

    # Convert y-ticks to percentages.
    ax.yaxis.set_major_formatter(pct_formatter())

    # Set axes labels.
    ax.set_xlabel("P/Sales")
    ax.set_ylabel("Annualized Return")

    # Show grid.
    ax.grid()

    if print_stats:
        print_statistics(model=model, psales_t=x, ann_rets=y)
    
    return ax


def plot_ann_returns_multi(years, years_range=0, filename=None,
                           figsize=None, *args, **kwargs):
    """
    Create plot with multiple sub-plots from `plot_ann_returns`
    for different years and years_range.
    
    :param years: List of years.
    :param years_range: Either integer or list of integers.
    :param filename: Full path to save figure to disk.
    :param figsize: Tuple with the figure-size.
    :return: Matplotlib Figure
    """
    
    # Number of sub-plots to create, one for each year.
    n = len(years)

    # Ensure `years_range` is a list or numpy array.
    if not isinstance(years_range, (list, np.ndarray)):
        years_range = np.repeat(years_range, repeats=n)

    # Figure size.
    if figsize is None:
        figsize = (10, 12.5 * n / 3)
        
    # Create new plot with sub-plots.
    fig, axs = plt.subplots(nrows=n, figsize=figsize)
    
    # Create each of the sub-plots.
    for ax, y, y_range in zip(axs, years, years_range):
        plot_ann_returns(ax=ax, years=y, years_range=y_range,
                         *args, **kwargs)
    
    # Adjust padding.
    fig.tight_layout()
    
    # Save plot to a file?
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

    return fig


def print_statistics(model, psales_t, ann_rets):
    """
    Calculate and print the Goodness of Fit statistics
    for a model's forecast compared to the baseline.

    The p-value results from a paired t-test whether
    the values are equal. A p-value close to zero means
    that the values are unlikely to be equal.

    :param model:
        Instance of the ForecastModel class.

    :param psales_t:
        Array with different P/Sales ratios at buy-time.

    :param ann_rets:
        Array with the corresponding annualized returns.
    """
    
    # Print header.
    print("\tForecast\tBaseline\tp-value")
    print("=================================================")

    # Mean Absolute Error (MAE).
    mae_forecast, mae_baseline, p_value = model.MAE(psales_t=psales_t,
                                                    ann_rets=ann_rets)
    msg = "MAE:\t{0:.1%}\t\t{1:.1%}\t\t{2:.2e}"
    msg = msg.format(mae_forecast, mae_baseline, p_value)
    print(msg)

    # Mean Squared Error (MSE).
    mse_forecast, mse_baseline, p_value = model.MSE(psales_t=psales_t,
                                                    ann_rets=ann_rets)
    msg = "MSE:\t{0:.2e}\t{1:.2e}\t{2:.2e}"
    msg = msg.format(mse_forecast, mse_baseline, p_value)
    print(msg)

    # R^2.
    R_squared = model.R_squared(psales_t=psales_t,
                                ann_rets=ann_rets)
    msg = "R^2:\t{0:.2f}"
    msg = msg.format(R_squared)
    print(msg)