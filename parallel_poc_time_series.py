import time
import pandas as pd
import numpy as np
from prophet import Prophet
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# function that generates random time-series between a time period:
def rnd_timeserie(min_date, max_date):
    time_index = pd.date_range(min_date, max_date)
    dates = pd.DataFrame({'ds': pd.to_datetime(time_index.values)}, index=range(len(time_index)))
    y = np.random.random_sample(len(dates)) * 10
    dates['y'] = y
    return dates


def run_prophet(timeserie):
    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.fit(timeserie)
    forecast = model.make_future_dataframe(periods=90, include_history=False)
    forecast = model.predict(forecast)
    return forecast


series = [rnd_timeserie('2018-01-01','2018-12-30') for x in range(0,500)]

# f = run_prophet(series[0])
# f.head()
#
# start_time = time.time()
# result = list(map(lambda timeserie: run_prophet(timeserie), tqdm(series)))
# print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    series = [rnd_timeserie('2018-01-01', '2018-12-30') for x in range(0, 500)]

    start_time = time.time()
    with Pool(cpu_count()) as p:
        predictions = list(tqdm(p.imap(run_prophet, series), total=len(series)))

    print("--- %s seconds ---" % (time.time() - start_time))
