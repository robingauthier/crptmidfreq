import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime, timedelta
from crptmidfreq.stepperc.incr_ewmkurt import EwmKurtStepper as EwmKurtStepperCython
from crptmidfreq.stepper.incr_ewmkurt import EwmKurtStepper as EwmKurtStepperNumba


def compare_speed_numba_cython():
    # Generate test data
    np.random.seed(42)
    n_samples = 20_000_000
    n_codes = 20

    # Create datetime range
    base_dt = datetime(2023, 1, 1)
    dt = np.array([base_dt + timedelta(minutes=i) for i in range(n_samples)], dtype='datetime64')

    # Create random codes and values
    dscodes = np.random.randint(0, n_codes, size=n_samples)
    values = np.random.randn(n_samples)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'datetime': dt,
        'dscode': dscodes,
        'value': values
    })

    time1 = pd.to_datetime('now')
    # Calculate using our implementation
    stepper = EwmKurtStepperCython(window=100)
    result = stepper.update(df['datetime'].values, df['dscode'].values, df['value'].values)
    time2 = pd.to_datetime('now')
    stepper = EwmKurtStepperNumba(window=100)
    result = stepper.update(df['datetime'].values, df['dscode'].values, df['value'].values)
    time3 = pd.to_datetime('now')

    dtimeCython = time2-time1
    dtimeNumba = time3-time2
    pprint({
        'cython': dtimeCython,
        'numba': dtimeNumba,
    })

    # for 1M
    # {'cython': Timedelta('0 days 00:00:00.794260'),
    # 'numba': Timedelta('0 days 00:00:01.447199')}

    # for 20M
    # {'cython': Timedelta('0 days 00:00:16.007647'),
    # 'numba': Timedelta('0 days 00:00:06.333569')}


# ipython -i -m crptmidfreq.res.speed_comp
if __name__ == "__main__":
    compare_speed_numba_cython()
