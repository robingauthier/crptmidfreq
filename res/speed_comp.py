import numpy as np
import pandas as pd
from pprint import pprint
from datetime import datetime, timedelta


def compare_speed_numba_cython():
    # Generate test data
    np.random.seed(42)
    n_samples = 3_000_000
    n_codes = 100

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
    #stepper = EwmKurtStepperCython(window=100)
    stepper=EwmStepperCython(window=100)
    time1_1 = pd.to_datetime('now')
    result = stepper.update(df['datetime'].values, df['dscode'].values, df['value'].values)
    time2 = pd.to_datetime('now')
    #stepper = EwmKurtStepperNumba(window=100)
    stepper = EwmStepperNumba(window=100)
    result = stepper.update(df['datetime'].values, df['dscode'].values, df['value'].values)
    time3 = pd.to_datetime('now')

    dtimeCython = time2-time1
    dtimeNumba = time3-time2
    pprint({
        'cython': dtimeCython,
        'cython_init': time1_1-time1,
        'cython_run': time2-time1_1,
        'numba': dtimeNumba,
    })


# ipython -i -m crptmidfreq.res.speed_comp
if __name__ == "__main__":
    compare_speed_numba_cython()
