import numpy as np
import pandas as pd
from crptmidfreq.utils.common import clean_folder
from ..incr_pivot import PivotStepper
from ..incr_unpivot import UnPivotStepper

# pytest ./stepper/tests/test_incr_unpivot.py --pdb --maxfail=1

def test_pivot_stepper_update():
    clean_folder('test_pfp')
    stepper = PivotStepper(folder='test_pfp')
    stepper2 = UnPivotStepper(folder='test_pfp')
    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int64)
    serie = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)

    df1 = pd.DataFrame({'dt':dt,'dscode':dscode,'serie':serie})
    dt2,nd = stepper.update(dt, dscode, serie)
    
    
    ndt,ndscode,nserie=stepper2.update(dt2,nd)
    df2 = pd.DataFrame({'dt':ndt,'dscode':ndscode,'nserie':nserie})
    df = df1.merge(df2,on=['dt','dscode'],how='inner')
    np.testing.assert_array_almost_equal(df['serie'].values, df['nserie'].values)
    