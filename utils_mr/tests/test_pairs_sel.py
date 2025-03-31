import pytest
import numpy as np
import pandas as pd

# If your pick_k_pairs is in a different module, update the import as needed:
# from your_module_name import pick_k_pairs
from crptmidfreq.utils_mr.pairs_sel import pick_k_pairs

# pytest ./crptmidfreq/utils_mr/tests/test_pairs_sel.py --pdb --maxfail=1
# /Users/sachadrevet/src/crptmidfreq/utils_mr/tests/test_pairs_sel.py


def test_pick_k_pairs():
    # Hardcoded input data
    dscode1 = np.array([1, 1, 1, 2, 2, 3, 4, 5], dtype=np.int64)
    dscode2 = np.array([2, 3, 4, 3, 4, 4, 5, 6], dtype=np.int64)
    distance = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)

    k = 2

    # Call the function
    featd = pick_k_pairs(dscode1, dscode2, distance, k)
    rdf = pd.DataFrame(featd)

    assert rdf.loc[lambda x:(x['dscode1'] == 1) & (x['dscode2'] == 2)].shape[0] > 0
    assert rdf.loc[lambda x:(x['dscode1'] == 4) & (x['dscode2'] == 5)].shape[0] > 0
    assert rdf.shape[0] == 5
