import numpy as np
import pandas as pd
from crptmidfreq.utils.common import clean_folder
from ..incr_pivot import PivotStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_pivot.py --pdb --maxfail=1


def test_pivot_stepper_update():
    clean_folder('test_pfp')
    stepper = PivotStepper(folder='test_pfp')

    dt = np.array([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
    dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int64)
    serie = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)

    udts, nd = stepper.update(dt, dscode, serie)
    nd = dict(nd)
    nd['dt'] = udts
    rdf = pd.DataFrame(nd).set_index('dt')
    cdf = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie
    })\
        .pivot_table(index='dt', columns='dscode', values='serie')
    assert rdf.equals(cdf)


def test_pivot_stepper_update2():
    # what happens if the universe increases?

    clean_folder('test_pfp')
    stepper = PivotStepper(folder='test_pfp')
    c = 8
    dt = np.array([1, 2, 2, 3, 4, 4, 6, 6, 8, 8, 8, 11], dtype=np.int64)
    dscode = np.array([1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 3, 2], dtype=np.int64)
    serie = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)

    udts1, nd1 = stepper.update(dt[:c], dscode[:c], serie[:c])
    nd1 = dict(nd1)
    stepper.save()
    stepper2 = PivotStepper(folder='test_pfp')
    udts2, nd2 = stepper2.update(dt[c:], dscode[c:], serie[c:])
    nd2 = dict(nd2)
    udts = np.concatenate([udts1, udts2], axis=0)

    for k in nd2.keys():
        if k not in nd1.keys():
            nd1[k] = np.zeros(nd1[1].shape[0])
            nd1[k][:] = np.nan

    nd = {k: np.concatenate([nd1[k], nd2[k]], axis=0) for k in nd1.keys()}

    nd['dt'] = udts
    rdf = pd.DataFrame(nd).set_index('dt')
    cdf = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie
    })\
        .pivot_table(index='dt', columns='dscode', values='serie')
    assert rdf.equals(cdf)


def test_pivot_stepper_update3():
    clean_folder('test_pfp')
    stepper = PivotStepper(folder='test_pfp')
    c = 8
    dt = np.array([1, 2, 2, 3, 4, 4, 6, 6,   
                   8, 8, 8, 11,11,11], dtype=np.int64)
    dscode = np.array([1, 1, 2, 1, 1, 2, 1, 2, 
                       3, 4, 5, 3,4,5], dtype=np.int64)
    serie = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)

    udts1, nd1 = stepper.update(dt[:c], dscode[:c], serie[:c])
    nd1 = dict(nd1)
    stepper.save()
    stepper2 = PivotStepper(folder='test_pfp')
    udts2, nd2 = stepper2.update(dt[c:], dscode[c:], serie[c:])
    nd2 = dict(nd2)
    udts = np.concatenate([udts1, udts2], axis=0)

    for k in nd2.keys():
        if k not in nd1.keys():
            nd1[k] = np.zeros(nd1[1].shape[0])
            nd1[k][:] = np.nan

    nd = {k: np.concatenate([nd1[k], nd2[k]], axis=0) for k in nd1.keys()}

    nd['dt'] = udts
    rdf = pd.DataFrame(nd).set_index('dt')
    cdf = pd.DataFrame({
        'dt': dt,
        'dscode': dscode,
        'serie': serie
    })\
        .pivot_table(index='dt', columns='dscode', values='serie')
    assert rdf.equals(cdf)
