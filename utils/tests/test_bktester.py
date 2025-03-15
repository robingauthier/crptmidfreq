import pandas as pd
import numpy as np

from ..bktester import  bktest_stats
# pytest ./utils/tests/test_bktester.py --pdb --maxfail=1

def test_bktest_stats():
    # Create a date range (as pd.DatetimeIndex)
    dt = pd.date_range(start="2020-01-01", periods=20, freq='D').to_numpy()
    # Create a sample P&L series; for index-based filtering later, it is a pd.Series.
    ypred = np.linspace(1, 20, 20)
    y = np.linspace(-10, 10, 20)
    dscodes = np.int64(np.ones(20))
    lots = np.ones(20)
    
    # Run the function.
    results = bktest_stats(dt, dscodes,ypred,y,lots)
    assert results['sr']>0
    print(pd.Series(results))

