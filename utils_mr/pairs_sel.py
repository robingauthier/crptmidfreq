import numpy as np
from numba import njit
from numba.typed import Dict, List
from numba.core import types


@njit
def pick_k_pairs_loc(dscode1, dscode2, distance, k,
                     rdscode1, rdscode2, rdistance):
    """
    Greedily pick pairs (dscode1[i], dscode2[i]) in ascending distance order,
    subject to the constraint that each stock appears in exactly k pairs at most.

    Parameters
    ----------
    dscode1 : int64[:]  (array of stock codes)
    dscode2 : int64[:]  (array of stock codes)
    distance : float64[:]  (array of distances, same length as dscode1/dscode2)
    k : int64  (max number of times any given stock can appear)

    Returns
    -------
    selected_pairs : List of (stockA, stockB) tuples (int64)
        The chosen pairs in ascending distance order.
    """

    n = distance.shape[0]
    # 1) Sort all pairs by ascending distance
    idx_sorted = np.argsort(distance)

    # 2) We'll track usage counts per stock in a Numba Dict
    usage_count = Dict.empty(key_type=types.int64, value_type=types.int64)

    # 4) Loop over pairs from smallest to largest distance
    for idx in idx_sorted:
        s1 = dscode1[idx]
        s2 = dscode2[idx]
        dst = distance[idx]
        if s1 == s2:
            continue
        # Check usage_count in usage dict
        # if not present, it means 0 so far
        if s1 not in usage_count:
            usage_count[s1] = 0
        if s2 not in usage_count:
            usage_count[s2] = 0

        # We only pick this pair if both s1 and s2 haven't reached k usage
        if usage_count[s1] < k and usage_count[s2] < k:
            # Add pair
            if s1 < s2:
                rdscode1.append(s1)
                rdscode2.append(s2)
            else:
                rdscode1.append(s2)
                rdscode2.append(s1)
            rdistance.append(dst)

            # Increment usage
            usage_count[s1] = usage_count[s1] + 1
            usage_count[s2] = usage_count[s2] + 1


def pick_k_pairs(dscode1, dscode2, distance, k):
    rdscode1 = List.empty_list(types.int64)
    rdscode2 = List.empty_list(types.int64)
    rdistance = List.empty_list(types.float64)
    pick_k_pairs_loc(dscode1, dscode2, distance, k,
                     rdscode1, rdscode2, rdistance)
    featd = {
        'dscode1': rdscode1,
        'dscode2': rdscode2,
        'dist': rdistance,
    }
    return featd
