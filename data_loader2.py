import pandas as pd
import os
from data_keys import *

data_dir = 'data/'


def _resample_daily(data):
    """
    Пересэмплирование данных с использованием линейной интерполяции.

    :param data: DataFrame или Series Pandas.
    :return: Пересэмплированные данные с ежедневной частотой.
    """
    return data.resample('D').interpolate(method='linear')


def _load_data(path):
    """
    Загружает CSV-файл с табуляцией в качестве разделителя, индексом дат в первом столбце
    и форматом даты MM/DD/YYYY.

    Это простой обертка над Pandas.read_csv().

    :param path: Путь к файлу с данными.
    :return: DataFrame Pandas.
    """
    data = pd.read_csv(path,
                       sep="\t",
                       index_col=0,
                       parse_dates=True,
                       dayfirst=False)

    return data


def load_price_yahoo(ticker):
    """
    Загрузка данных о цене акций из CSV-файла Yahoo.

    Загружаются только цены 'Close' и 'Adj Close',
    которые интерполируются до ежедневных значений.

    Данные о ценах 'Close' корректируются с учетом сплитов акций.

    Данные о ценах 'Adj Close' корректируются как для сплитов акций,
    так и для дивидендов, что соответствует общей доходности (Total Return).

    https://help.yahoo.com/kb/SLN2311.html

    :param ticker: Тикер для загрузки данных.
    :return: DataFrame Pandas с данными SHARE_PRICE и TOTAL_RETURN.
    """

    # Путь для подгрузки данных
    path = os.path.join(data_dir, ticker + " Share-Price (Yahoo).csv")

    # Считываем стоимость акции с файла
    price_raw = pd.read_csv(path,
                            index_col=0,
                            header=0,
                            sep=',',
                            parse_dates=[0],
                            dayfirst=False)

    # Переименовываем столбцы
    columns = \
        {
            'Adj Close': TOTAL_RETURN,
            'Close': SHARE_PRICE
        }
    price = price_raw.rename(columns=columns)

    # Выбираем столбцы, которые нам нужны
    price = price[[TOTAL_RETURN, SHARE_PRICE]]

    # Интерполируем для поучения стоимостей за все дни
    price_daily = _resample_daily(price)

    return price_daily


def load_earnings_per_share(ticker, df, profit_margin=True):
    """
    Загружает данные о прибыли на акцию (Earnings Per Share) из файла и добавляет их в DataFrame.
    Также рассчитываются коэффициент P/E и рентабельность продаж(profit margin).

    :param ticker:
        Название акции, используемое в именах файлов, например, "WMT".

    :param df:
        DataFrame Pandas с данными SHARE_PRICE.

    :param profit_margin:
        Булевый параметр, указывающий, добавлять ли рентабельность продаж в DataFrame.
        Требует, чтобы в df уже были данные SALES_PER_SHARE.

    :return:
        None. Данные добавляются в DataFrame `df`.
    """

    # Подгружаем данные
    path = os.path.join(data_dir, ticker + " Earnings Per Share.txt")
    earnings_per_share = _load_data(path=path)

    # Добавлем в DataFrame (интерполированные по дням)
    df[EARNINGS_PER_SHARE] = _resample_daily(earnings_per_share)

    # Добавляем мультипликатор (По дням)
    df[PE] = df[SHARE_PRICE] / df[EARNINGS_PER_SHARE]

    # Добавляем рентабельность продаж (По дням)
    if profit_margin:
        df[PROFIT_MARGIN] = df[EARNINGS_PER_SHARE] / df[SALES_PER_SHARE]


def load_sales_per_share(ticker, df):
    """
    Загружает данные о продажах на акцию (Sales Per Share) из файла и добавляет их в DataFrame.
    Также рассчитываются коэффициент P/Sales и годовой рост продаж на акцию.

    :param ticker:
        Название акции, используемое в именах файлов, например, "WMT".

    :param df:
        DataFrame Pandas с данными SHARE_PRICE.

    :return:
        None. Данные добавляются в DataFrame `df`.
    """

    # Подгружаем данные
    path = os.path.join(data_dir, ticker + " Sales Per Share.txt")
    sales_per_share = _load_data(path=path)

    # Добавляем к DataFrame (Интерполированный по дням).
    df[SALES_PER_SHARE] = _resample_daily(sales_per_share)

    # Добавляем мультипликатор (По дням)
    df[PSALES] = df[SHARE_PRICE] / df[SALES_PER_SHARE]

    # Добавляем годовой рост дохода (По дням).
    # Мы нарушаем наше правило, что для момента t мы считаем её будущее значение, 
    # в то время как тут считаем для текущего значени (заглядывая назад).
    # Но на самом деле это не повлияет на расчёт, так как мы всё равно берём матожу,
    # А ей вообще по барабану на то в какой последовательности идут данные
    df[SALES_GROWTH] = df[SALES_PER_SHARE].pct_change(periods=365, fill_method=None)


def load_book_value_per_share(ticker, df):
    """
    Загружает данные о балансовой стоимости на акцию (Book-Value Per Share) из файла и добавляет их в DataFrame.
    Также рассчитывается коэффициент P/Book.

    :param ticker:
        Название акции, используемое в именах файлов, например, "WMT".

    :param df:
        DataFrame Pandas с данными SHARE_PRICE.

    :return:
        None. Данные добавляются в DataFrame `df`.
    """

    # Подгружаем данные
    path = os.path.join(data_dir, ticker + " Book-Value Per Share.txt")
    book_value_per_share = _load_data(path=path)

    # Добавляем к DataFrame (Интерполированный по дням).
    df[BOOK_VALUE_PER_SHARE] = _resample_daily(book_value_per_share)

    # Добавляем мультипликатор (По дням)
    df[PBOOK] = df[SHARE_PRICE] / df[BOOK_VALUE_PER_SHARE]


def load_dividend_TTM(ticker, df):
    """
    Загружает данные о дивиденде на акцию за последние двенадцать месяцев (TTM, Trailing Twelve Months) из файла и
    добавляет их в DataFrame. Также рассчитываются некоторые связанные коэффициенты оценки.

    :param ticker:
        Название акции или индекса, используемое в именах файлов, например, "S&P 500".

    :param df:
        DataFrame Pandas с данными SHARE_PRICE.

    :return:
        None. Данные добавляются в DataFrame `df`.
    """

    # Подгружаем данные
    path = os.path.join(data_dir, ticker + " Dividend Per Share TTM.txt")
    dividend_per_share_TTM = _load_data(path=path)

    # Добавляем к DataFrame (Интерполированный по дням).
    df[DIVIDEND_TTM] = _resample_daily(dividend_per_share_TTM)

    # Добавляем мультипликаторы (По дням)
    df[PDIVIDEND] = df[SHARE_PRICE] / df[DIVIDEND_TTM]
    # Вот тут есть некоторого рода ошибка:
    # Мы нарушаем наше правило, что для момента t мы считаем её будущее значение, 
    # в то время как тут считаем для текущего значени (заглядывая назад).
    # Но на самом деле это не повлияет на расчёт, так как мы всё равно берём матожу,
    # А ей вообще по барабану на то в какой последовательности идут данные
    df[DIVIDEND_YIELD] = df[DIVIDEND_TTM] / df[SHARE_PRICE]


def load_stock_data(ticker, earnings=True, sales=True, book_value=True,
                    dividend_TTM=False):
    """
    Загружает данные для одной акции из нескольких файлов
    и объединяет их в один DataFrame Pandas.

    - Цена загружается из файла Yahoo.
    - Остальные данные загружаются из отдельных файлов.

    Общая доходность берется напрямую из данных о ценах Yahoo.
    Оценочные коэффициенты, такие как P/E и P/Sales, рассчитываются ежедневно
    из интерполированных данных.

    :param ticker:
        Название акции, используемое в именах файлов, например, "WMT".

    :param earnings:
        Булевый параметр, указывающий, загружать ли файл с данными о прибыли на акцию (EPS).

    :param sales:
        Булевый параметр, указывающий, загружать ли файл с данными о продажах на акцию.

    :param book_value:
        Булевый параметр, указывающий, загружать ли файл с данными о балансовой стоимости на акцию.

    :param dividend_TTM:
        Булевый параметр, указывающий, загружать ли файл с данными о дивидендах на акцию за 12 месяцев (TTM).

    :return: DataFrame Pandas с данными.
    """
    # Подгружаем данные из файлов
    price_daily = load_price_yahoo(ticker=ticker)

    # Используем DataFrame для цен, чтобы затем добавить больше колонок к нему
    df = price_daily

    # Оставляем только строки, где стоимость акции определена
    df.dropna(subset=[SHARE_PRICE], inplace=True)

    # Подгружаем доход к одной акции 
    if sales:
        load_sales_per_share(ticker=ticker, df=df)

    # Подгружаем прибыль к одной акции
    # Для расчета рентабельности продаж требуется также загрузить данные о продажах на акцию (Sales Per Share).
    if earnings:
        load_earnings_per_share(ticker=ticker, df=df)

    # Подгружаем Book-Value Per Share data
    if book_value:
        load_book_value_per_share(ticker=ticker, df=df)

    # Подгружаем суму дивиденды за 12 месяцев на одну акцию
    if dividend_TTM:
        load_dividend_TTM(ticker=ticker, df=df)

    return df