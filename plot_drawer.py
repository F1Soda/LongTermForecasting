from simfin.names import (TICKER, DATE)

from itertools import combinations
import seaborn as sns


# Строки, содержащие имена для данных графиков
WINDOW_LENGTH = 'Window Length'
MEAN_RETURN = 'Mean Daily Return'
STD_RETURN = 'Std.Dev. Daily Return'
CORR = 'Correlation'
TICKER2 = 'Ticker2'
TICKER_PAIR = 'Ticker Pair'



def plot_rolling_stats(df, stat_name, tickers, windows,
                       col_ticker=TICKER, y_dashed_line=None):
    """
    Строит графики скользящей статистики для нескольких тикеров акций
    или пар тикеров. Для каждого тикера или пары создается
    отдельный подграфик в своей строке.
    

    :param df:
        Pandas DataFrame с данными для построения графиков.
        Генерируется функцией `calc_rolling_stats`.

    :param stat_name:
        Строка с названием статистики.

    :param tickers:
        Список строк с тикерами акций.

    :param windows:
        Список целых чисел с длинами окон.

    :param col_ticker:
        Строка с названием столбца в `df`, используемого
        для разделения графиков по строкам. Например, TICKER или TICKER_PAIR.

    :param y_dashed_line:
        Если не `None`, рисует горизонтальную черную пунктирную
        линию на этом значении.

    :return:
        Объект оси Matplotlib.
    """

    # Строим сколзящие статистики в подграфике для каждого тикера
    facet_kws = {'sharey': False, 'sharex': True}
    g = sns.relplot(x=DATE, y=stat_name, hue=WINDOW_LENGTH,
                    row=col_ticker, data=df, kind='line',
                    facet_kws=facet_kws, height=4, aspect=10/4)

    # Добавлять горизонтальную черную пунктирную линию?
    if y_dashed_line is not None:
        for ax in g.axes:
            # Рисуем черную пунктирную линию
            ax[0].axhline(y=y_dashed_line, color='black', linestyle='dashed')

    # Поправляем отступы
    g.tight_layout()

    return g


def draw_return_statatistics_plots(df_mean, df_std, df_corr, tickers, windows):
    # Конвертируем df_mean, чтобы его мог отобразить Seaborn
    id_vars = [DATE, TICKER]
    df_mean = df_mean.reset_index().melt(id_vars=id_vars,
                                         var_name=WINDOW_LENGTH,
                                         value_name=MEAN_RETURN)

    # Конвертируем df_std, чтобы его мог отобразить Seaborn
    df_std = df_std.reset_index().melt(id_vars=id_vars,
                                       var_name=WINDOW_LENGTH,
                                       value_name=STD_RETURN)

    # Комбинируем два столбца с тикерами в одну пару тикеров
    df_corr = df_corr.reset_index()
    df_corr[TICKER_PAIR] = list(zip(df_corr[TICKER], df_corr[TICKER2]))
    df_corr = df_corr.drop(columns=[TICKER, TICKER2])
    
    ticker_pairs = list(combinations(tickers, r=2))

    # Используем строки, которые имеют релевантные пары тикеров
    mask = df_corr[TICKER_PAIR].isin(ticker_pairs)
    df_corr = df_corr.loc[mask]

    # Конвертируем df_corr, чтобы его мог отобразить Seaborn
    df_corr = df_corr.melt(id_vars=[DATE, TICKER_PAIR],
                           var_name=WINDOW_LENGTH, value_name=CORR)
    
    # Рисуем скользящее среднее
    plot_rolling_stats(df=df_mean, stat_name=MEAN_RETURN,
                       col_ticker=TICKER, tickers=tickers,
                       windows=windows, y_dashed_line=1.0)

    # Рисуем скользящее среднее квадартичное отклонение
    plot_rolling_stats(df=df_std, stat_name=STD_RETURN,
                       col_ticker=TICKER, tickers=tickers,
                       windows=windows, y_dashed_line=0.0)

    # Рисуем скользящюю корреляцию
    plot_rolling_stats(df=df_corr, stat_name=CORR,
                       col_ticker=TICKER_PAIR, tickers=tickers,
                       windows=windows, y_dashed_line=0.0)   