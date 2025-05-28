from itertools import chain, combinations
import simfin as sf
from simfin.names import (TOTAL_RETURN, CLOSE, VOLUME, TICKER,
                          PSALES, DATE)
from simfin.utils import BDAYS_PER_YEAR
import numpy as np


def one_period_returns(prices, future):
    """
    Рассчитывает доходность за один период для заданных цен акций.
    Этот метод считает доходность по следующей формуле:

    R_t = (X_t - X_{t-1}) / X_{t-1}
    
    Обратите внимание, что к этим значениям добавляется 1.0, например:
    1.05 означает прирост на 5% за один период, а 0.98 — убыток на 2%.

    :param prices:
        Pandas DataFrame, содержащий, например, дневные цены акций.
        
    :param future:
        Логическое значение, указывающее, нужно ли вычислять доходность в будущем (True)
        или в прошлом (False).
        
    :return:
        Pandas DataFrame с доходностью за один период.
    """
    # Однопериодная доходность + 1
    # Изначально формула pct_change = (X_t - X_{t-1}) / X_{t-1}
    rets = prices.pct_change(periods=1) + 1.0
    
    # Сдвигаем на 1 шаг назад, чтобы получить формулу для будущей доходности
    if future:
        rets = rets.shift(-1)

    return rets

def get_bad_tickers(daily_returns_all, all_tickers, df_daily_prices):
    # Находим тикеры, чьи ежедневная рыночкая капитализация меньше 1_000_000 
    daily_trade_mcap = df_daily_prices[CLOSE] * df_daily_prices[VOLUME]
    mask = (daily_trade_mcap.groupby(level=0).median() < 1e7)
    bad_tickers1 = mask[mask].reset_index()[TICKER].unique()

    # Находим тикеры, чьё максимальный доход в день составлял больше 100%
    mask2 = (daily_returns_all > 2.0)
    mask2 = (np.sum(mask2) >= 1) # если есть хотя бы один такой день, то убираем
    bad_tickers2 = mask2[mask2].index.to_list()

    # Находим тикеры, у которых больше 20% процентов данных nan-ы(Not-a-Number)
    mask3 = (daily_returns_all.isna().sum(axis=0) > 0.2 * len(daily_returns_all))
    bad_tickers3 = mask3[mask3].index.to_list()

    # Находим тикеры, которые заканчиваются '_old'.
    # Они были удалены по какой то причине
    bad_tickers4 = [ticker for ticker in all_tickers[1:]
                    if ticker.endswith('_old')]

    # Тикеры, для которых мы знаем, что есть проблемы с данными
    bad_tickers5= ['FCAUS']

    # Объединяем всё вместе
    bad_tickers = np.unique(np.concatenate([bad_tickers1, bad_tickers2,
                                            bad_tickers3, bad_tickers4,
                                            bad_tickers5]))
    
    return bad_tickers

def prepare_data(df_daily_prices):
    # Используем ежедневные данные "Total Return", которые представляют собой цены акций,
    # скорректированные на сплиты акций и реинвестирование дивидендов.
    # Это объект Pandas DataFrame в виде матрицы, где строки соответствуют временным шагам,
    # а столбцы — отдельным тикерам акций.
    daily_prices = df_daily_prices[TOTAL_RETURN].unstack().T

    # Удаляем строки, содержащие очень мало данных. Иногда в этом наборе данных
    # встречаются «фантомные» точки данных для некоторых акций, например, по выходным.
    num_stocks = len(daily_prices.columns)
    daily_prices = daily_prices.dropna(thresh=int(0.1 * num_stocks))

    # Убираем первый ряд, так как иногда в нем отсутсвуют данные
    daily_prices = daily_prices.iloc[:, 1:]

    # Ежедневная доходность акций, рассчитанная на основе "Total Return".
    # Мы могли бы использовать функцию hub.returns() из SimFin,
    # но этот код позволяет вам проще использовать другой источник данных.
    # Это объект Pandas DataFrame в виде матрицы, где строки — это временные шаги,
    # а столбцы — это отдельные тикеры акций.
    daily_returns_all = one_period_returns(prices=daily_prices, future=True)

    # Убираем пустые строчки(Должен быть самый последний)
    daily_returns_all = daily_returns_all.dropna(how='all')

    # Получаем список всех доступных тикеров
    all_tickers = daily_prices.columns.to_list()

    bad_tickers = get_bad_tickers(daily_returns_all, all_tickers, df_daily_prices)

    # Получаем список валидных тикеров
    valid_tickers = [x for x in all_tickers if x not in bad_tickers] 

    # ВАЖНО!
    # Здесь выполняется заполнение пропущенных данных вперёд и назад (forward- и backward-fill)
    # для ежедневных цен акций, чтобы все акции имели данные за одни и те же даты.
    # Иначе при создании портфеля из множества случайных акций нам пришлось бы
    # ограничивать период данными, которые есть у всех акций одновременно.
    # Например, для портфеля из 300 акций общий период с данными
    # может быть всего несколько лет.
    # Когда пропущенные цены акций заполняются таким образом,
    # это соответствует тому, что часть портфеля инвестируется в кэш.
    # Это не влияет на портфели Buy&Hold, Threshold, Adaptive
    # и Adaptive+, но может повлиять на портфель Rebalanced.
    # Вы можете легко отключить это заполнение и использовать только
    # исходные данные о ценах акций во всех экспериментах ниже,
    # просто изменив условие на `if False` вместо `if True`.
    if True:
        # Заполняем все nan-ы используя сначала forward-заполнение, а затем backward
        daily_prices = daily_prices.ffill().bfill()
        
        # Перерасчитываем дневную доходность.
        # Для дней, которые мы заполнили значениями с прошлого
        # доходность будет 0% 
        daily_returns_all = one_period_returns(prices=daily_prices, future=True)
        
    # Используем данные только для валидных тикеров
    daily_returns = daily_returns_all[valid_tickers]

    # убираем последню строчку, так как там только нан
    daily_returns = daily_returns.iloc[:-1]    

    return daily_returns
