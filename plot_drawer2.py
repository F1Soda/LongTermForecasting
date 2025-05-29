import pandas as pd
import numpy as np
from data_keys import *
import matplotlib.pyplot as plt
from forecast_model import ForecastModel
from matplotlib.ticker import FuncFormatter


def pct_formatter(num_decimals=0):
    """Форматтер процентов, используемый при построении графиков."""
    return FuncFormatter(lambda x, _: '{:.{}%}'.format(x, num_decimals))


def annualized_returns(series, years):
    """
    Вычисляет annualized доходности для всех возможных
    периодов заданной длины в годах.

    Например, имея annualized доходность акции, мы хотим
    узнать annualized доходность в период длиной 10 лет.

    :param series:
        Pandas Series, например, с совокупной доходностью акции(Total Return).
        Предполагается, что данные представлены по дням.

    :param years:
        Количество лет в периоде.

    :return:
        Pandas Series той же длины, что и исходная серия. Для каждого
        дня указана среднегодовая доходность для периода, начинающегося
        в этот день и продолжающегося заданное количество лет.
        Конец серии содержит значения NA для заданного количества лет.
    """

    # Количество дней для сдвига в данных. Все года имеют 365 дней
    # кроме высокосных (366), которые случаются каждые 4 года.
    # Поэтому в среднем в году 365.25 дней.
    days = int(years * 365.25)

    # Рассчитываем annualized доходности для всех периодов данной длины
    # Замечание: важно чтобы данные были представлены по дням, иначе
    # сдвиг на 365 делал смещение на большее число дней, так как в 
    # основном в году представлены около 250 дней (в выходные нет данных).
    ann_return = (series.shift(-days) / series) ** (1 / years) - 1.0

    return ann_return


def prepare_ann_returns(df, years, key=PSALES):
    """
    Подготовливаем данные по annualized доходностям.
    Ось x задается ключом (например, PSALES), а ось y — annualized доходностью.

    :param df:
        Pandas DataFrame с колонками, названными как key и TOTAL_RETURN.
    :param years:
        Количество лет для расчета среднегодовой доходности.
    :param key:
        Название колонки данных для оси x, например, PSALES или PBOOK.
    :return:
        (x, y) — Pandas Series с key и скорректированной ANN_RETURN.
    """

    # Создаем новый DataFrame, чтобы не изменять оригинальный.
    # По сути, мы используем его для синхронизации данных,
    # которые нас интересуют, по общим датам и для исключения NA-значений.
    df2 = pd.DataFrame()

    # Копируем key данные, например, P/Sales
    df2[key] = df[key]

    # Расчитываем annualized доходности для периода в years лет,
    # используя Total Return
    df2[ANN_RETURN] = annualized_returns(series=df[TOTAL_RETURN], years=years)

    # Убираем все строчки с nan-ами
    df2.dropna(axis=0, how='any', inplace=True)

    # Возвращем результат
    x = df2[key]
    y = df2[ANN_RETURN]

    return x, y


def plot_ann_returns(ticker, df, years,
                     dividend_yield=None, sales_growth=None,
                     psales=None, psales_min=None, psales_max=None,
                     ax=None, print_stats=False):
    """
    Создаем график с историческими доходностями,
    отображающими зависимость коэффициента P/Sales от будущих
    среднегодовых доходностей.
    
    Затем Накладываем на этот график оценку среднего значения и стандартного
    отклонения из математической прогностической модели.

    Необязательные параметры берутся из DataFrame `df`, если они
    не переданы явно. Это позволяет переопределить некоторые или
    все данные, используемые в прогностической модели, например,
    для изменения предположений о будущем росте продаж.
    
    :param ticker: Строка с тикером акции или индекса.
    :param df: Pandas DataFrame.
    :param years: Количество лет инвестирования.
    :param dividend_yield: (Необязательно) Массив дивидендной доходности.
    :param sales_growth: (Необязательно) Массив годового роста продаж.
    :param psales: (Необязательно) Массив коэффициентов P/Sales.
    :param psales_min: (Необязательно) Минимальное значение P/Sales для построения кривых.
    :param psales_max: (Необязательно) Максимальное значение P/Sales для построения кривых.
    :param ax: (Необязательно) Объект Matplotlib Axis для графика.
    :return: None
    """
    
    # Выбираем данные, которые нам нужны
    df2 = df[[TOTAL_RETURN, DIVIDEND_YIELD, SALES_GROWTH, PSALES]]

    # Убираем строчки с потерянными данными
    df2 = df2.dropna()
    
    # Создаем новый график, если ось графика plt не предоставлена
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(211)

    # Часть заголовка
    start_date, end_date = df2.index[[0, -1]]
    title_dates = "({0}-{1})".format(start_date.year, end_date.year)

    # Получаем annualized доходы из исторических данных
    x, y = prepare_ann_returns(df=df2, years=years, key=PSALES)

    # Первая часть заголовка графика
    title1 = "[{0}] {1}-Year Ann. Return {2}".format(ticker, years, title_dates)
    
    # Метка для точек графика, обозначающие настоящие доходности
    label_returns = "Actual Returns"

    # Получаем Dividend Yield если его нету
    if dividend_yield is None:
        dividend_yield = df2[DIVIDEND_YIELD]

    # Получаем Sales Growth если его нету
    if sales_growth is None:
        sales_growth = df2[SALES_GROWTH]

    # Получаем P/Sales если его нету
    if psales is None:
        psales = df2[PSALES]

    # Получаем минимальный P/Sales если его нету
    if psales_min is None:
        psales_min = np.min(psales)
    
    # Получаем максимальный P/Sales если его нету
    if psales_max is None:
        psales_max = np.max(psales)
        
    # Создаем нашу модель и обучаем на данных
    model = ForecastModel(dividend_yield=dividend_yield,
                          sales_growth=sales_growth,
                          psales=psales, years=years)

    # Равномерно распределённый P/Sales для графика
    psales_t = np.linspace(start=psales_min, stop=psales_max, num=100)

    # Используем модель чтобы предсказать среднее и стандартное отклонение
    mean, std = model.forecast(psales_t=psales_t)

    # Рисуем матожидание annualized доходности
    # Так же приводим значение R^2 чтобы понять как хорошо модель описывает реальность
    R_squared = model.R_squared(psales_t=x, ann_rets=y)
    label = "Forecast Mean (R^2 = {0:.2f})".format(R_squared)
    ax.plot(psales_t, mean, color="black", label=label)
    
    # Рисуем диапазон в стандратное отклонение
    color = "green"
    alpha = 0.3
    # Рисуем линии ниже и выше среднего
    ax.plot(psales_t, mean-std, color=color,
            label="Forecast Mean $\pm$ 1 Std.Dev.")
    ax.plot(psales_t, mean+std, color=color)
    # Заполняем область прозрачным цветом
    ax.fill_between(psales_t, mean+std, mean-std,
                    color=color, edgecolor=color, alpha=alpha)
    
    # Рисуем диапазон в два стандартных отклонения
    color = "red"
    alpha = 0.1
    # Рисуем линии ниже и выше среднего.
    ax.plot(psales_t, mean-2*std, color=color,
            label="Forecast Mean $\pm$ 2 Std.Dev.")
    ax.plot(psales_t, mean+2*std, color=color) 
    # Заполняем область прозрачным цветом
    ax.fill_between(psales_t, mean-std, mean-2*std,
                    color=color, edgecolor=color, alpha=alpha)
    ax.fill_between(psales_t, mean+std, mean+2*std,
                    color=color, edgecolor=color, alpha=alpha)

    # Рисуем по точкам график с фактическими значениями P/Sales против среднегодовой доходности.
    # Каждая точка окрашивается в цвет в зависимости от даты (позиции в массиве).
    # Точки растеризуются (преобразуются в пиксели) для экономии места
    # при сохранении в векторный графический файл.
    n = len(x)
    c = np.arange(n) / n
    ax.scatter(x, y, marker='o', c=c, cmap='plasma',
               label=label_returns, rasterized=True)

    # Рисуем исторический средний как горизонтальную пунктирную линию
    y_mean = np.mean(y)
    label = 'Actual Mean = {0:.1%}'.format(y_mean)
    ax.axhline(y=y_mean, color="black", linestyle=":", label=label)

    # Отображаем метки всего того, что нариосвали ранее
    ax.legend()

    # Создаем заголовок графика
    # Вторая часть заголовка. Выписываем формулу для среднего
    msg = "E[Ann Return] = {0:.2f} / (P/Sales ^ (1/{1})) - 1"
    title2 = msg.format(model.a, years)
    # Вторая часть заголовка. Выписываем формулу для стандратного отклонения
    msg = "Std[Ann Return] = {0:.3f} / (P/Sales ^ (1/{1}))"
    title3 = msg.format(model.b, years)
    # Combine and set the plot-title.
    title = "\n".join([title1, title2, title3])
    ax.set_title(title)

    # Конвертируем доходность в проценты
    ax.yaxis.set_major_formatter(pct_formatter())

    # Устанавливаем метки на осях
    ax.set_xlabel("P/Sales")
    ax.set_ylabel("Annualized Return")

    # Показываем сетку
    ax.grid()

    # Выписываем статистику (MSE, MAE, R^2, p-value)
    if print_stats:
        print_statistics(model=model, psales_t=x, ann_rets=y)
    
    return ax


def print_statistics(model, psales_t, ann_rets):
    """
    Вычислить и вывести статистику качества аппроксимации (Goodness of Fit)
    для прогноза модели по сравнению с историческим средним.

    p-значение рассчитывается с помощью парного t-теста для проверки
    гипотезы о неравенстве значений. p-значение, близкое к нулю, указывает
    на то, что, скорее всего, наша модель лучше описывает историчесие данные

    :param model:
        Экземпляр класса ForecastModel.

    :param psales_t:
        Массив различных коэффициентов P/Sales на момент покупки.

    :param ann_rets:
        Массив соответствующих среднегодовых доходностей.
    """
    
    # Печатаем заголовок
    print("\tForecast\tBaseline\tp-value")
    print("=================================================")

    # Средняя абсолютная ошибка (MAE).
    mae_forecast, mae_baseline, p_value = model.MAE(psales_t=psales_t,
                                                    ann_rets=ann_rets)
    msg = "MAE:\t{0:.1%}\t\t{1:.1%}\t\t{2:.2e}"
    msg = msg.format(mae_forecast, mae_baseline, p_value)
    print(msg)

    # Средняя квадратичная ошибка (MSE).
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