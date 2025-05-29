from scipy.stats import ttest_rel
import numpy as np

class ForecastModel:
    """
    Математическая модель, используемая для предсказания доходов на длительном промежутке
    """

    def __init__(self, dividend_yield, sales_growth,
                 psales, years):
        """
        Создаем новую модель и обучаем ее.
            
        :param dividend_yield: Массив с дивидендными доходностями.
        :param sales_growth: Массив с годовыми темпами роста продаж.
        :param psales: Массив с коэффициентами P/Sales.
        :param years: Количество лет для расчета годовой доходности.
        """
        
        # Копируем аргументы в self.
        # Обратите внимание на +1 для дивидендной доходности и роста продаж,
        # чтобы не выполнять эту операцию несколько раз ниже.
        self.dividend_yield = np.array(dividend_yield) + 1
        self.sales_growth = np.array(sales_growth) + 1
        self.psales = psales
        self.years = years
        
        # Расчитываем параметр `a` для матожидания ann.return.
        self.a = self.mean_parameter()

        # Расчитываем параметр `b` для стандартного квадратичного отклонения ann.return.
        self.b = self.std_parameter()

    def forecast(self, psales_t):
        """
        Используем обученную модель для прогнозирования матожидания значения и стандартного отклонения
        будущих доходностей акций.

        :param psales_t: Массив с различными значениями P/Sales на момент покупки.
        :return: Два массива — со средним значением и годовым стандартным отклонением 
                для годовых доходностей по каждому значению psales_t.
        """

        # В переводе на годовой показатель psales_t
        psales_t_ann = psales_t ** (1/self.years)

        # Предсказываем матожидание и стандартное квадратичное отклоненеи для ann. returns
        # для различных значениях P/Sales на момент покупки акции
        mean = self.a / psales_t_ann - 1.0
        std = self.b / psales_t_ann

        return mean, std

    def mean_parameter(self):
        """
        Оцениваем параметр `a`, используемый в формуле для
        среднегодовой доходности, на основе массивов с распределениями
        дивидендной доходности, роста продаж и P/Sales.

        :return: Параметр `a` для формулы средней доходности.
        """

        # Предполагается, что dividend_yield и sales_growth уже с +1
        a = np.mean(self.dividend_yield) \
          * np.mean(self.sales_growth) \
          * np.mean(self.psales ** (1/self.years))

        return a

    def std_parameter(self, num_samples=10000):
        """
        Оцениваем параметр `b`, используемый в формуле для
        стандартного отклонения среднегодовой доходности, на основе массивов
        с распределениями дивидендной доходности, роста продаж и P/Sales.

        Оценка производится методом Монте-Карло / повторной выборки
        предоставленных данных, которые считаются независимыми друг от друга
        и во времени.

        :param num_samples: Количество выборок Монте-Карло.
        :return: Параметр `b` для формулы стандартного отклонения доходности.
        """
        
        # Мы также могли бы вычислить параметр b, используя данные
        # и формулу напрямую, но для этого требуется, чтобы все
        # массивы данных имели одинаковую длину.
        # Для исторических данных результаты очень
        # похожи на результаты моделирования Монте-Карло.
        # return np.std( self.dividend_yield * self.sales_growth * self.psales ** (1/self.years) )

        # Размер массивов для семплирования
        shape = (num_samples, self.years)
        num_samples_total = np.prod(shape)

        # Сэмплируем dividend yield. Предполагается что он уже с +1.
        dividend_yield_sample = np.random.choice(self.dividend_yield, size=shape)
        # Складываем рост в течении нескольки лет 
        dividend_yield_sample = np.prod(dividend_yield_sample, axis=1)

        # Сэмплируем sales-growth. Предполагается что он уже с +1.
        sales_growth_sample = np.random.choice(self.sales_growth, size=shape)
        # Складываем рост в течении нескольки лет 
        sales_growth_sample = np.prod(sales_growth_sample, axis=1)

        # Сэмплируем P/Sales в момент покупки
        psales_sample = np.random.choice(self.psales, size=num_samples)

        # Комбинируем
        combined_sample = dividend_yield_sample * sales_growth_sample * psales_sample

        # Считаем параметр b
        b = np.std(combined_sample ** (1/self.years))

        return b
    
    def _ttest(self, err_forecast, err_baseline):
        """
        Выполняем t-тест на ошибках нашей модели
        и средним, чтобы оценить, чья ошибка меньше.
        
        Когда полученное p_value близко к нулю, среднее ошибка нашей модели
        скорее всего меньше чем ошибка прострого среднего.

        И наоборот, если p_value = 1, то наша модель проигрывает простому среднему 

        :param err_forecast:
            Ошибки нашей модели.

        :param err_baseline:
            Ошибки среднего.

        :return:
            p_value
        """
        
        # Парный т-тест.
        t_value, p_value = ttest_rel(a=err_forecast, b=err_baseline, alternative="less")

        return p_value

    def MAE(self, psales_t, ann_rets):
        """
        Вычисляем среднюю абсолютную ошибку (MAE) между прогнозируемым annualized
        значением модели и фактическими annualized доходностями.

        Также вычисляет MAE между средним и фактическими annualized доходностями.

        Также вычисляет p-значение для проверки гипотезы о неравенстве MAE
        прогнозной и среднего.

        :param psales_t:
            Массив различных коэффициентов P/Sales на момент покупки.
            
        :param ann_rets:
            Массив соответствующих annualized доходностей.

        :return:
            mae_forecast: MAE между прогнозом модели и фактическими доходностями.
            mae_baseline: MAE между средним и фактическими доходностями.
            p_value: p-value для проверки неравенства MAE двух моделей.
        """


        # Предсказываем матожидание и стандартное среднеквадратичное отклонение доходностей,
        # для исторических значений P/Sales.
        mean_forecast, std_forecast = self.forecast(psales_t=psales_t)

        # Ошибка между реальными данными и моделью.
        err_forecast = np.abs(ann_rets - mean_forecast)
        
        # Ошибка между простым историческим средним и реальными данными
        err_baseline = np.abs(ann_rets - np.mean(ann_rets))
        
        # Средняя абсолютная ошибка (MAE).
        mae_forecast = np.mean(err_forecast)
        mae_baseline = np.mean(err_baseline)
        
        # Проверка гипотезы, что MAE нашей модели выше чему у среднего (H_0)
        p_value = self._ttest(err_forecast=err_forecast,
                              err_baseline=err_baseline)

        return mae_forecast, mae_baseline, p_value
    
    def MSE(self, psales_t, ann_rets):
            """
            Вычисляем среднюю квадратичную ошибку (MSE) между прогнозируемым annualized
            значением модели и фактическими annualized доходностями.

            Также вычисляет MSE между средним и фактическими annualized доходностями.

            Также вычисляет p-значение для проверки гипотезы о неравенстве MSE
            прогнозной и среднего.

            :param psales_t:
                Массив различных коэффициентов P/Sales на момент покупки.
                
            :param ann_rets:
                Массив соответствующих annualized доходностей.

            :return:
                mse_forecast: MSE между прогнозом модели и фактическими доходностями.
                mse_baseline: MSE между средним и фактическими доходностями.
                p_value: p-value для проверки неравенства MAE двух моделей.
            """

            # Предсказываем матожидание и стандартное среднеквадратичное отклонение доходностей,
            # для исторических значений P/Sales.
            mean_forecast, std_forecast = self.forecast(psales_t=psales_t)

            # Ошибка между реальными данными и моделью.
            err_forecast = (ann_rets - mean_forecast) ** 2
            
            # Ошибка между простым историческим средним и реальными данными
            err_baseline = (ann_rets - np.mean(ann_rets)) ** 2
            
            # Средне квадратичная ошибка (MSE).
            mse_forecast = np.mean(err_forecast)
            mse_baseline = np.mean(err_baseline)

            # Проверка гипотезы, что MSE нашей модели выше чему у среднего (H_0)
            p_value = self._ttest(err_forecast=err_forecast,
                                err_baseline=err_baseline)

            return mse_forecast, mse_baseline, p_value
    
    def R_squared(self, psales_t, ann_rets):
        """
        Вычисляет коэффициент детерминации R^2 для оценки качества
        аппроксимации между прогнозируемым средним значением и
        фактическими annualized доходностями.

        Значение R^2, равное единице, означает идеальное совпадение,
        и прогнозная модель объясняет всю дисперсию в данных.
        Значение R^2, равное нулю, означает, что модель не объясняет
        никакой дисперсии в данных.
        
        Обратите внимание, что поскольку модель прогноза является нелинейной,
        значение R^2 может быть отрицательным, если модель плохо
        подходит для данных с большой дисперсией.

        :param psales_t:
            Массив различных коэффициентов P/Sales на момент покупки.
            
        :param ann_rets:
            Массив соответствующих среднегодовых доходностей.

        :return:
            Значение R^2.
        """

        # Предсказываем матожидание и стандартное среднеквадратичное отклонение доходностей,
        # для исторических значений P/Sales.
        mean_forecast, std_forecast = self.forecast(psales_t=psales_t)

        # Ошибка между реальными данными и моделью.
        err_forecast = (ann_rets - mean_forecast) ** 2
        
        # Ошибка между простым историческим средним и реальными данными
        err_baseline = (ann_rets - np.mean(ann_rets)) ** 2
        
        # Сумма квадратов ошибок (SSE) for для модели
        sse = np.sum(err_forecast)
        
        # Сумма квадратов ошибок (SST) для исторического среднего
        sst = np.sum(err_baseline)

        # Значение R^2
        R_squared = 1.0 - sse / sst
        
        return R_squared