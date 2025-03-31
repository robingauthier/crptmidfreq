
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from numba.typed import Dict
from numba.typed import List
from numba.core import types
from numba import njit
from crptmidfreq.utils_mr.pairs_sel import pick_k_pairs_loc


class PairsSelKStepper(BaseStepper):

    def __init__(self, folder='', name='', k=3):
        """
        """
        super().__init__(folder, name)
        self.k = k

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, k=3):
        """Load instance from saved state or create new if not exists"""
        return PairsSelKStepper.load_utility(cls,
                                             folder=folder,
                                             name=name,
                                             k=k)

    def update(self, dts, dscode1, dscode2, distance):
        """
        pserie : pivoted serie
        puniv : pivoted PIT universe with 1/0

        Returns:
        resultd : numba Dict dscode: array of residuals
        result_D : 2 dim array with the diagonal values of the matrix D
        """
        rdts = List.empty_list(types.int64)
        rdscode1 = List.empty_list(types.int64)
        rdscode2 = List.empty_list(types.int64)
        rdistance = List.empty_list(types.float64)
        n = dts.shape[0]

        # we need to mark when dt has same value
        fi = 0  # first i for current dt
        li = 0  # last i for current dt
        current_dt = dts[0]

        for i in range(n):
            dt = dts[i]
            if (dt != current_dt) or (i == n-1):
                li = i
                nb1 = len(rdscode1)
                pick_k_pairs_loc(dscode1[fi:li],
                                 dscode2[fi:li],
                                 distance[fi:li],
                                 self.k,
                                 rdscode1,
                                 rdscode2,
                                 rdistance)
                nb2 = len(rdscode1)
                for t in range(nb2-nb1):
                    rdts.append(current_dt)

                # resetting now
                fi = min(n-1, i)
                current_dt = dts[fi]

        featd = {
            'dtsi': rdts,
            'dscode1': rdscode1,
            'dscode2': rdscode2,
            'dist': rdistance,
        }
        return featd
