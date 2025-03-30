import numpy as np
import pandas as pd
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.stepper.incr_pivot import PivotStepper
from crptmidfreq.stepper.incr_unpivot import UnPivotStepper
from crptmidfreq.stepper.incr_svd import SVDStepper
from crptmidfreq.stepper.tests.test_utils import generate_data

# pytest ./crptmidfreq/stepper/tests/test_incr_svd.py --pdb --maxfail=1


def test_svd_stepper_update():
    clean_folder('test_pfp')

    dt = np.array([1, 1, 1, 1,
                   3, 3, 3, 3,
                   6, 6, 6, 6,
                   9, 9, 9, 9,
                   11, 11, 11, 11], dtype=np.int64)
    dscode = np.array([1, 2, 3, 4,
                       1, 2, 3, 4,
                       1, 2, 3, 4,
                       1, 2, 3, 4,
                       1, 2, 3, 4], dtype=np.int64)
    serie = np.array([0.1, 0.12, 0.09, 0.11,
                      -0.2, -0.25, -0.3, -0.29,
                      0.0, 0.1, -0.01, 0.01,
                      0.7, 0.6, 0.8, 0.7,
                      -0.7, -0.6, -2.2, -0.75], dtype=np.float64)

    stepper = PivotStepper(folder='test_pfp')
    dt2, nd = stepper.update(dt, dscode, serie)

    stepper2 = SVDStepper(folder='test_pfp2', lookback=3, fitfreq=1, n_comp=2)
    result_resid, result_D = stepper2.update(dt2, nd)

    rdf = pd.DataFrame(dict(result_resid))
    rd = pd.DataFrame(result_D)

    # first component must explain most of the risk
    assert (rd[0]/rd[1]).mean() > 6

    assert rdf.shape[1] == 4


def test_svd_stepper_2():
    clean_folder('test_pfp')

    c = 1000
    featd = generate_data(ndts=300, ndscode=20, rank=2)

    dt = featd['dtsi']
    dscode = featd['dscode']
    serie = featd['serie']
    univ = featd['univ']

    stepper = PivotStepper(folder='test_pfp')
    dt2, seried = stepper.update(dt, dscode, serie)

    stepperU = PivotStepper(folder='test_pfp', name='univ')
    dt2, univd = stepper.update(dt, dscode, univ)

    stepper2 = SVDStepper(folder='test_pfp', lookback=50, fitfreq=2, n_comp=2)
    result_resid, result_D = stepper2.update(dt2, seried, univd)

    stepper3 = UnPivotStepper(folder='test_pfp', name='univ3')
    ndt, ndscode, nserie = stepper3.update(dt2, result_resid)

    df1 = pd.DataFrame(featd)
    df2 = pd.DataFrame({
        'dtsi': ndt,
        'dscode': ndscode,
        'serie2': nserie,
    })
    df = df1.merge(df2, on=['dtsi', 'dscode'], how='outer')
    corr_original = df[df['univ'] > 0][['serie', 'serie2']].corr().iloc[0, 1]

    corr = df[df['univ'] > 0][['serie_res', 'serie2']].corr().iloc[0, 1]
    assert corr > 0.7
