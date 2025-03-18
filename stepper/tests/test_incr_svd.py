import numpy as np
import pandas as pd
from crptmidfreq.utils.common import clean_folder
from ..incr_pivot import PivotStepper
from ..incr_svd import SVDStepper

# pytest ./stepper/tests/test_incr_svd.py --pdb --maxfail=1

def test_svd_stepper_update():
    clean_folder('test_pfp')
    

    dt = np.array([1, 1, 1, 1,
                   3, 3, 3, 3,
                   6, 6, 6, 6,
                   9, 9, 9,9,
                   11,11,11,11], dtype=np.int64)
    dscode = np.array([1, 2, 3, 4,
                       1, 2, 3, 4,
                       1, 2, 3, 4,
                       1, 2, 3, 4,
                       1, 2, 3, 4], dtype=np.int64)
    serie = np.array([0.1, 0.12, 0.09,0.11,
                      -0.2, -0.25, -0.3, -0.29,
                      0.0, 0.1, -0.01, 0.01,
                      0.7, 0.6, 0.8,0.7,
                      -0.7, -0.6, -0.8,-0.75], dtype=np.float64)

    stepper = PivotStepper(folder='test_pfp')
    dt2,nd = stepper.update(dt, dscode, serie)
    
    stepper2 = SVDStepper(folder='test_pfp2',lookback=3,fitfreq=1,n_comp=2)
    result_resid,result_D,result_FR=stepper2.update(dt2,nd)
    
    rdf=pd.DataFrame(result_resid)
    rfr = pd.DataFrame(result_FR)
    rd = pd.DataFrame(result_D)
    print(rfr)
    assert rfr.iloc[2,0]>0
    assert rfr.iloc[3,0]>0
    assert rfr.iloc[4,0]<0
    assert np.all(rfr.iloc[3:,0].abs()>=rfr.iloc[3:,1].abs())
    
    # To be fair we need to perform more checks    
