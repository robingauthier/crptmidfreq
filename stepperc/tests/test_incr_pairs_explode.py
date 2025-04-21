import numpy as np
import pandas as pd

from crptmidfreq.stepper.incr_pairs_explode import PairsExplodeStepper
from crptmidfreq.utils.common import clean_folder

# pytest ./crptmidfreq/stepper/tests/test_incr_pairs_explode.py --pdb --maxfail=1


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
    serie = np.float64(dscode)

    sdts = np.array([2, 2, 9, 9])
    sdscode1 = np.array([1, 1, 2, 4])
    sdscode2 = np.array([2, 3, 3, 3])

    m = PairsExplodeStepper(folder=folder)
    featd = m.update(dts, dscode, serie, sdts, sdscode1, sdscode2)
    rdf = pd.DataFrame(featd)

    ed = {'dts': {0: 2, 1: 3, 2: 3, 3: 4, 4: 4, 5: 6, 6: 6, 7: 9, 8: 11, 9: 11},
          'dscode1': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 4},
          'dscode2': {0: 2, 1: 2, 2: 3, 3: 2, 4: 3, 5: 2, 6: 3, 7: 3, 8: 3, 9: 3},
          'serie1': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 2.0, 8: 2.0, 9: np.nan},
          'serie2': {0: 2.0, 1: 2.0, 2: np.nan, 3: 2.0, 4: 3.0, 5: 2.0, 6: 3.0, 7: 3.0, 8: 3.0, 9: 3.0}}
    edf = pd.DataFrame(ed)
    assert rdf.equals(edf)
