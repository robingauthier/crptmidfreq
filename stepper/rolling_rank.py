import numpy as np
from numba import njit

from stepper.rolling_base import RollingStepper


@njit
def update_rolling_rank(timestamps,
                        dscode,
                        values,
                        position,
                        rolling_dict,
                        last_timestamps,
                        window):
    """

    """
    result = np.zeros(len(dscode), dtype=np.float64)
    for i in range(len(dscode)):
        code = dscode[i]
        value = values[i]
        ts = timestamps[i]

        # Check timestamp is increasing for this code
        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:  # Changed from <= to < to allow same timestamp
            raise ValueError("DateTime must be strictly increasing per code")

        if code not in position:
            position[code] = 0

        if code not in rolling_dict:
            rolling_dict[code] = np.empty(window, dtype=np.float64)
            for j in range(window):
                rolling_dict[code][j] = np.nan

        position_loc = position[code]
        rolling_dict[code][position_loc] = value
        last_timestamps[code] = ts

        # ranking is done with argsort().argsort() -- twice !
        result[i] = np.argsort(rolling_dict[code]).argsort()[position_loc]
        result[i]=(result[i]-window/2)/window
        position[code] = (position_loc + 1) % window
    return result


class RollingRankStepper(RollingStepper):
    def update(self, dt, dscode, values):
        if len(dscode) != len(values):
            raise ValueError("Codes and values arrays must have the same length")
        if len(dt) != len(values):
            raise ValueError("Codes and values arrays must have the same length")

        if not dt.dtype == 'int64':
            timestamps = dt.astype('datetime64[ns]').astype('int64')
        else:
            timestamps = dt

        res = update_rolling_rank(timestamps, dscode, values,
                                  self.position, self.values, self.last_ts, self.window)
        return res

### Below should move in the test folder

def generate_data(n_samples, n_codes):
    """Generate test data"""
    np.random.seed(42)

    # Generate random codes
    dscode = np.random.randint(0, n_codes, n_samples)

    # Generate random series
    serie = np.random.randn(n_samples)

    # Generate increasing datetime
    base = np.datetime64('2024-01-01')
    dt = np.array([base + np.timedelta64(i, 'm') for i in range(n_samples)])

    return dt, dscode, serie


def example_against_pandas():
    import pandas as pd
    # Generate test data
    n_samples = 1000
    n_codes = 10
    window = 10
    dt, dscode, serie = generate_data(n_samples, n_codes)

    # Create and run EwmStepper on first half
    ewm = RollingRankStepper(folder='test_data', name='test_ewm', window=window)
    seriec = ewm.update(dt, dscode, serie)

    # Create pandas DataFrame for comparison
    df = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie,
        'seriec': seriec,

    })

    # Calculate pandas EWM
    df['serier'] = df.groupby('dscode')['serie'].transform(
        lambda x: x.rolling(window=window).rank()
    )

    # Compare results using correlation
    correlation = df['serier'].corr(df['seriec'])
    print(f"Correlation between pandas and implementation: {correlation}")
    assert correlation > 0.9, f"Expected correlation >0.9, got {correlation}"

    return True
# ipython -i -m stepper.rolling_rank
if __name__=='__main__':
    example_against_pandas()