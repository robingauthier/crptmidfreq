
from crptmidfreq.stepper.incr_bktest import *


# pytest ./crptmidfreq/stepper/tests/test_incr_bktest.py --pdb --maxfail=1

def generate_data(n_samples, n_codes):
    """Generate test data"""
    np.random.seed(42)

    # Generate random codes
    dscode = np.random.randint(0, n_codes, n_samples)

    # Generate random series
    rand1 = np.random.randn(n_samples)
    rand2 = np.random.randn(n_samples)
    rand3 = np.random.randn(n_samples)

    # Generate increasing datetime
    base = np.datetime64('2024-01-01')
    dt = np.sort(np.random.randint(0, 200, n_samples))
    wgt = np.ones(n_samples)
    featd = {
        'dtsi': dt,
        'dscode': dscode,
        'forward_fh1': rand3+rand1,
        'sig_1': rand1,
        'sig_2': rand2,
        'wgt': wgt
    }
    return featd


def test_against_pandas():
    # Generate test data
    n_samples = 10000
    n_codes = 10
    window = 5
    featd = generate_data(n_samples, n_codes)

    c = int(n_samples/2)
    # Create and run EwmStepper on first half
    ewm = BktestStepper(folder='test_data1', name='test_ewm')
    ewm.update(featd)
    r1 = ewm.display_stats()

    ewm = BktestStepper(folder='test_data2', name='test_ewm')
    ewm.update({k: v[:c] for k, v in featd.items()})
    ewm.save()
    ewm.update({k: v[c:] for k, v in featd.items()})
    ewm.save()
    r2 = ewm.display_stats()

    assert abs(r1['cnt'].mean()-r2['cnt'].mean()) < 5
    assert abs(r1['rpt'].mean()-r2['rpt'].mean()) < 10
