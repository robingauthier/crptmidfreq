import pandas as pd

from crptmidfreq.stepper.incr_distance_correl import CorrelDistanceStepper
from crptmidfreq.stepper.incr_pivot import PivotStepper
from crptmidfreq.stepper.tests.test_utils import generate_data
from crptmidfreq.utils.common import clean_folder

# pytest ./crptmidfreq/stepper/tests/test_incr_distance_correl.py --pdb --maxfail=1


def test_dst_stepper():
    clean_folder('test_pfp')

    c = 1000
    featd = generate_data(ndts=300, ndscode=20, rank=2)

    dt = featd['dtsi']
    dscode = featd['dscode']
    serie = featd['serie']

    stepper = PivotStepper(folder='test_pfp')
    dt2, seried = stepper.update(dt, dscode, serie)

    stepper2 = CorrelDistanceStepper(folder='test_pfp', lookback=50, fitfreq=2)
    rfeatd = stepper2.update(dt2, seried)

    df = pd.DataFrame(rfeatd)
    dfo = pd.DataFrame(featd)
    corrm = dfo.pivot_table(index='dtsi', columns='dscode', values='serie').fillna(0.0).corr()
    assert df.shape[0] > 0

    dscode1 = 3
    dscode2 = 4
    avg_corr = df[(df['dscode1'] == dscode1) & (df['dscode2'] == dscode2)]['dist'].mean()
    true_corr = corrm.loc[dscode1, dscode2]
    assert abs(avg_corr-true_corr) < 0.3
