import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from numba.typed import Dict
from numba.core import types
from crptmidfreq.stepper.incr_pairs_sel_k import PairsSelKStepper
from crptmidfreq.utils.common import clean_folder

# pytest ./crptmidfreq/stepper/tests/test_incr_pairs_sel_k.py --pdb --maxfail=1


def test_pairs_sel_k():
    folder = 'test_pairs'
    clean_folder(folder)
    dts = np.array([1, 1, 1,
                    1, 1, 1,

                    3, 3, 3, 3,
                    3, 3, 3,
                    3, 3, 3,

                    5, 5, 5,
                    5, 5, 5])
    dscode1 = np.array([1, 1, 1,
                       2, 2, 3,

                       1, 1, 1, 1,
                       2, 2, 2,
                       3, 3, 4,

                       1, 1, 1,
                       2, 2, 3,])
    dscode2 = np.array([2, 3, 4,
                       2, 3, 4,

                       2, 3, 4, 5,
                       3, 4, 5,
                       4, 5, 5,

                       2, 3, 4,
                       3, 4, 4])

    distance = np.arange(dscode1.shape[0])

    m = PairsSelKStepper(folder=folder, k=2)
    featd = m.update(dts, dscode1, dscode2, distance)
    rdf = pd.DataFrame(featd)
    ed = {'dtsi': {0: 1, 1: 1, 2: 1, 3: 3, 4: 3, 5: 3, 6: 3, 7: 5, 8: 5, 9: 5},
          'dscode1': {0: 1, 1: 1, 2: 2, 3: 1, 4: 1, 5: 2, 6: 4, 7: 1, 8: 1, 9: 2},
          'dscode2': {0: 2, 1: 3, 2: 3, 3: 2, 4: 3, 5: 3, 6: 5, 7: 2, 8: 3, 9: 3},
          'dist': {0: 0.0, 1: 1.0, 2: 4.0, 3: 6.0, 4: 7.0, 5: 10.0, 6: 15.0, 7: 16.0, 8: 17.0, 9: 19.0}}
    edf = pd.DataFrame(ed)
    assert rdf.equals(edf)
