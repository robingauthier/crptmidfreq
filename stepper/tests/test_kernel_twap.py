import numpy as np
import pandas as pd

from stepper.kernel_twap import RollingKernelTwapStepper
from utils.common import clean_folder


def test_rolling_kernel_twap():
    # Hardcoded test data
    dt = np.array([1, 2, 3, 5, 8, 9, 10, 15, 16, 17, 18, 20, 24, 26, 30, 35, 36, 37, 42, 45], dtype=np.int64)
    dscode = np.array([1] * 20, dtype=np.int64)
    values = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, ], dtype=np.float64)

    stepper = RollingKernelTwapStepper(window=20, spike_interval=10, half_life=20, spike_width=0.2)
    result = stepper.update(dt, dscode, values, dt)
    pd.Series(result).to_csv('~/Downloads/res.csv')

    # Expected values - Adjust based on kernel and logic
    assert np.abs(result[18] - 1) < 1e-3
    assert np.abs(result[0] - 1) < 1e-3
    assert np.abs(result[6] - 1) < 1e-3
    assert np.abs(result[3]) < 1e-3


def test_rolling_kernel_twap_save_load():
    # Hardcoded test data
    dt = np.array([1, 2, 3, 5, 8, 9, 10, 15, 16, 17, 18, 20, 24, 26, 30, 35, 36, 37, 42, 45], dtype=np.int64)
    dscode = np.array([1] * 20, dtype=np.int64)
    values = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, ], dtype=np.float64)

    c = 10
    clean_folder('test_kernel_twap1')
    clean_folder('test_kernel_twap2')
    stepper = RollingKernelTwapStepper.load(folder='test_kernel_twap1', window=20,
                                            spike_interval=10,
                                            half_life=20,
                                            spike_width=0.2)
    result = stepper.update(dt, dscode, values, dt)

    stepper1 = RollingKernelTwapStepper.load(folder='test_kernel_twap2', window=20,
                                             spike_interval=10,
                                             half_life=20,
                                             spike_width=0.2)
    result1 = stepper1.update(dt[:c], dscode[:c], values[:c], dt[:c])
    stepper2 = RollingKernelTwapStepper.load(folder='test_kernel_twap2')
    result2 = stepper2.update(dt[c:], dscode[c:], values[c:], dt[c:])

    df = pd.DataFrame({'r': result, 'rc': np.concatenate([result1, result2])})
    correl = df.corr().iloc[0, 1]
    assert correl > 0.9
