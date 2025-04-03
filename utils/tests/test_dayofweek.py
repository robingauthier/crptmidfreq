
import numpy as np
import pandas as pd
from crptmidfreq.utils.common import get_day_of_week_unix

# pytest ./crptmidfreq/utils/tests/test_dayofweek.py --pdb --maxfail=1


# Test function to compare results with pandas
def test_day_of_week():
    # Example: An array of timestamps in milliseconds since the epoch
    time_ms_array = np.array([1683043200000 + i*3600*24*1000 for i in range(60)])  # Example timestamps

    # Convert milliseconds to seconds
    time_s_array = time_ms_array // 1000

    # Convert milliseconds array to pandas day of week (0 = Monday, 6 = Sunday)
    pandas_day_of_week = pd.to_datetime(time_ms_array, unit='ms').dayofweek

    # Calculate the day of week using our function
    our_day_of_week = get_day_of_week_unix(time_s_array)

    # Ensure that our calculation matches pandas
    print(f"Pandas Day of Week: {pandas_day_of_week}")
    print(f"Our Day of Week: {our_day_of_week}")

    assert np.array_equal(pandas_day_of_week, our_day_of_week), "Test failed! Results do not match."
    print("Test passed! Results match.")
