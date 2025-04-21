import numpy as np
import pandas as pd

from crptmidfreq.stepper.incr_pairs_collapse import PairsCollapseStepper
from crptmidfreq.stepper.incr_pairs_explode import PairsExplodeStepper
from crptmidfreq.stepper.incr_unpivot import UnPivotStepper
from crptmidfreq.utils.common import clean_folder

# pytest ./crptmidfreq/stepper/tests/test_incr_pairs_collapse.py --pdb --maxfail=1

# By pairing stocks you cannot reconstruct the initial signal btw !!


def test_pairs_explode():
    folder = 'test_pairs'
    clean_folder(folder)
    dts = np.array([1, 1, 1,
                    2, 2, 2,
                    3, 3, 3,
                    4, 4, 4,
                    6, 6, 6,
                    9, 9, 9,
                    11, 11, 11, 11])
    dscode = np.array([1, 2, 3,
                       1, 2, 3,
                       1, 2, 4,
                       1, 2, 3,
                       1, 2, 3,
                       1, 2, 3,
                       1, 2, 3, 4])
    serie = np.array([0, -1, 1,
                      0, -2, 2,
                      0, -1.5, 1.5,
                      0, 1, -1,
                      0, 2, -2,
                      0, 3, -3,
                      1, -2, 0, 0])
    # Hard coding my pairs
    sdts = np.array([2])
    sdscode1 = np.array([1])
    sdscode2 = np.array([2])

    m = PairsExplodeStepper(folder=folder)
    featd = m.update(dts, dscode, serie, sdts, sdscode1, sdscode2)
    featd['seriediff'] = featd['serie1']-featd['serie2']
    featd['seriesum'] = featd['serie1']+featd['serie2']

    m2 = PairsCollapseStepper(folder=folder)
    dts2, pserie = m2.update(featd['dts'], featd['dscode1'], featd['dscode2'], featd['seriediff'])

    m3 = UnPivotStepper(folder=folder)
    ndt, ndscode, nserie = m3.update(dts2, pserie)

    rdf = pd.DataFrame({
        'dtsi': ndt,
        'dscode': ndscode,
        'serie_d': nserie,
    })
    edf = pd.DataFrame({
        'dtsi': dts,
        'dscode': dscode,
        'serie_o': serie,
    })

    df = rdf.merge(edf, on=['dtsi', 'dscode'], how='outer')
    assert df[~df['dscode'].isin([1, 2])]['serie_d'].fillna(0).abs().max() == 0.0
    assert df[df['dscode'].isin([1, 2])].groupby('dtsi')[['serie_d']].sum().abs().max().iloc[0] == 0.0
